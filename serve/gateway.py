#!/usr/bin/env python3
"""Loopback gateway for 4-byte big-endian lengths and opaque frame payloads.

One outstanding request per client and backend connection. Requests are assigned
FIFO to available persistent backend slots. No pickle loading, batching, or retry.
Python 3.9+ standard library only. Restart after draining clients to change ports.
"""
import argparse
import asyncio
import json
import logging
import signal
import struct
import time
from contextlib import suppress

LOG = logging.getLogger('gateway')
HEADER = struct.Struct('!I')
CHUNK = 64 * 1024
DEFAULT_PORTS = '10086-10093,10100-10107,10200-10207'


def parse_ports(spec):
    ports = set()
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            pieces = part.split('-')
            if len(pieces) != 2:
                raise ValueError(f'invalid port range: {part!r}')
            lo, hi = map(int, pieces)
            if not 1 <= lo <= hi <= 65535:
                raise ValueError(f'invalid port range: {part!r}')
            ports.update(range(lo, hi + 1))
        else:
            port = int(part)
            if not 1 <= port <= 65535:
                raise ValueError(f'port out of range: {port}')
            ports.add(port)
    if not ports:
        raise ValueError('at least one backend port is required')
    return sorted(ports)


async def close_writer(writer):
    if writer is not None:
        writer.close()
        with suppress(Exception):
            await asyncio.wait_for(writer.wait_closed(), 1.0)


class _Backend:
    def __init__(self, host, port):
        self.host, self.port = host, port
        self.reader = self.writer = None

    async def connect(self, timeout):
        if self.writer is None:
            self.reader, self.writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port, limit=CHUNK), timeout)

    async def close(self):
        writer, self.writer = self.writer, None
        self.reader = None
        await close_writer(writer)


class Gateway:
    def __init__(self, backend_ports, host='127.0.0.1', port=10900,
                 backend_host='127.0.0.1', max_frame=256*1024*1024,
                 io_timeout=30.0, queue_timeout=30.0, stats_file=None):
        if host not in ('127.0.0.1', '::1'):
            raise ValueError('gateway must bind 127.0.0.1 or ::1 only')
        if not 0 <= port <= 65535:
            raise ValueError('listen port out of range')
        self.host, self.port = host, port
        self.max_frame = int(max_frame)
        if not 1 <= self.max_frame <= 0xFFFFFFFF:
            raise ValueError('max_frame must be between 1 and 2^32-1')
        self.io_timeout, self.queue_timeout = float(io_timeout), float(queue_timeout)
        if self.io_timeout <= 0 or self.queue_timeout <= 0:
            raise ValueError('timeouts must be positive')
        ports = parse_ports(','.join(map(str, backend_ports)))
        self._backends = [_Backend(backend_host, p) for p in ports]
        self._available = asyncio.Queue()
        for backend in self._backends:
            self._available.put_nowait(backend)
        self.stats_file = stats_file
        self._stats = None
        self._clients = set()
        self._writers = set()
        self._closing = False
        self.server = None
        self.active_clients = 0
        self._client_count = self._request_count = 0

    async def start(self):
        if self.server is not None or self._closing:
            raise RuntimeError('gateway instances may only be started once')
        if self.stats_file:
            self._stats = open(self.stats_file, 'a', encoding='utf-8', buffering=1)
        try:
            self.server = await asyncio.start_server(self._accept, self.host, self.port, limit=CHUNK)
        except BaseException:
            if self._stats:
                self._stats.close()
                self._stats = None
            raise
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    def _accept(self, reader, writer):
        if self._closing:
            writer.close()
            return
        self._client_count += 1
        self.active_clients += 1
        self._writers.add(writer)
        task = asyncio.create_task(self._handle_client(reader, writer, self._client_count))
        self._clients.add(task)
        task.add_done_callback(self._clients.discard)

    async def close(self):
        self._closing = True
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        tasks = list(self._clients)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # A callback accepted immediately before shutdown may not have run its task.
        for writer in list(self._writers):
            await close_writer(writer)
        self._writers.clear()
        self.active_clients = 0
        await asyncio.gather(*(backend.close() for backend in self._backends))
        if self._stats:
            self._stats.close()
            self._stats = None

    async def _read_exact(self, reader, n):
        return await asyncio.wait_for(reader.readexactly(n), self.io_timeout)

    async def _drain(self, writer):
        await asyncio.wait_for(writer.drain(), self.io_timeout)

    async def _copy(self, reader, writer, size, row, stage, count_key):
        remaining = size
        while remaining:
            row['stage'] = stage + '_read'
            chunk = await self._read_exact(reader, min(CHUNK, remaining))
            row['stage'] = stage + '_write'
            writer.write(chunk)
            await self._drain(writer)
            row[count_key] += len(chunk)
            remaining -= len(chunk)

    async def _request(self, reader, writer, header, row):
        backend = None
        queue_started = backend_started = None
        try:
            row['request_bytes'] = HEADER.unpack(header)[0]
            if row['request_bytes'] > self.max_frame:
                raise ValueError(f'request frame exceeds max_frame={self.max_frame}')
            row['stage'] = 'backend_queue'
            queue_started = time.monotonic()
            try:
                backend = await asyncio.wait_for(self._available.get(), self.queue_timeout)
            finally:
                row['queue_wait_ms'] = (time.monotonic() - queue_started) * 1000
            row['backend_port'] = backend.port
            backend_started = time.monotonic()
            row['stage'] = 'backend_connect'
            await backend.connect(self.io_timeout)
            row['stage'] = 'backend_request_header'
            backend.writer.write(header)
            await self._drain(backend.writer)
            await self._copy(reader, backend.writer, row['request_bytes'], row,
                             'request_payload', 'request_forwarded_bytes')
            row['stage'] = 'backend_response_header'
            response_header = await self._read_exact(backend.reader, HEADER.size)
            row['response_bytes'] = HEADER.unpack(response_header)[0]
            if row['response_bytes'] > self.max_frame:
                raise ValueError(f'response frame exceeds max_frame={self.max_frame}')
            row['stage'] = 'client_response_header'
            writer.write(response_header)
            await self._drain(writer)
            await self._copy(backend.reader, writer, row['response_bytes'], row,
                             'response_payload', 'response_forwarded_bytes')
        except BaseException:
            # Even a partial request/response makes the backend stream unusable.
            # Never replay a request: inference may already have run.
            if backend:
                await backend.close()
            raise
        finally:
            if backend_started is not None:
                row['backend_latency_ms'] = (time.monotonic() - backend_started) * 1000
            if backend:
                self._available.put_nowait(backend)

    async def _handle_client(self, reader, writer, client_id):
        try:
            while True:
                # Idle connections are bounded by io_timeout, but time waiting for
                # the first header byte is excluded from request latency/metrics.
                try:
                    first = await self._read_exact(reader, 1)
                except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError):
                    return
                started = time.monotonic()
                self._request_count += 1
                row = dict(timestamp=time.time(), request_id=self._request_count,
                           client_id=client_id, request_bytes=None, response_bytes=None,
                           request_forwarded_bytes=0, response_forwarded_bytes=0,
                           queue_wait_ms=0.0, backend_latency_ms=0.0,
                           backend_port=None, error=None, error_stage=None,
                           stage='client_request_header')
                try:
                    header = first + await self._read_exact(reader, HEADER.size - 1)
                    await self._request(reader, writer, header, row)
                except asyncio.CancelledError:
                    row['error'] = 'CancelledError: gateway shutdown'
                    row['error_stage'] = row['stage']
                    raise
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                    row['error_stage'] = row['stage']
                    return
                finally:
                    row['total_latency_ms'] = (time.monotonic() - started) * 1000
                    row['active_clients'] = self.active_clients
                    row['bytes'] = row['request_forwarded_bytes'] + row['response_forwarded_bytes']
                    row.pop('stage')
                    for key in ('queue_wait_ms', 'backend_latency_ms', 'total_latency_ms'):
                        row[key] = round(row[key], 3)
                    self._record(row)
        finally:
            self.active_clients -= 1
            self._writers.discard(writer)
            await close_writer(writer)

    def _record(self, row):
        if self._stats:
            try:
                self._stats.write(json.dumps(row, separators=(',', ':')) + '\n')
            except OSError:
                LOG.exception('request metrics write failed')
        if row['error']:
            LOG.warning('request %s failed at %s (backend %s): %s', row['request_id'],
                        row['error_stage'], row['backend_port'], row['error'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listen-host', default='127.0.0.1', choices=['127.0.0.1', '::1'])
    parser.add_argument('--listen-port', type=int, default=10900)
    parser.add_argument('--backend-host', default='127.0.0.1')
    parser.add_argument('--backend-ports', default=DEFAULT_PORTS, help='comma-separated ports/ranges')
    parser.add_argument('--max-frame', type=int, default=256*1024*1024, help='maximum payload bytes')
    parser.add_argument('--io-timeout', type=float, default=30, help='seconds per connect/read/write, including idle clients')
    parser.add_argument('--queue-timeout', type=float, default=30, help='seconds waiting for an available backend')
    parser.add_argument('--stats-file', help='append JSONL per request, without payload contents')
    args = parser.parse_args(argv)
    try:
        ports = parse_ports(args.backend_ports)
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    async def run():
        gw = await Gateway(ports, host=args.listen_host, port=args.listen_port,
                           backend_host=args.backend_host, max_frame=args.max_frame,
                           io_timeout=args.io_timeout, queue_timeout=args.queue_timeout,
                           stats_file=args.stats_file).start()
        LOG.info('listening on %s:%s; backend ports=%s', gw.host, gw.port, ports)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        try:
            await stop.wait()
        finally:
            await gw.close()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
