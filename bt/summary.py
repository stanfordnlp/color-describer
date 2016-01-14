import atexit
import struct
import time

from tensorflow.core.util.event_pb2 import Event
from tensorflow.core.framework.summary_pb2 import Summary

from .crc32c import crc as crc32


class SummaryWriter(object):
    def __init__(self, filename, tick=5.0, max_queue_len=100):
        self.filename = filename

        self.tick = tick
        self.max_queue_len = max_queue_len

        self.last_append = time.time()
        self.queue = []

        # Truncate the file to start
        with open(filename, 'wb'):
            pass

        atexit.register(SummaryWriter.flush, self)

    def log_scalar(self, step, tag, val):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=val)])
        t = time.time()
        e = Event(wall_time=t, step=step, summary=summary)
        self.queue.append(e)
        if t - self.last_append >= self.tick or len(self.queue) >= self.max_queue_len:
            self.flush()
            self.last_append = t

    def flush(self):
        with open(self.filename, 'ab') as outfile:
            write_events(outfile, self.queue)
            del self.queue[:]


class SummaryReaderException(Exception):
    pass


def masked_crc(data):
    crc = crc32(data) & 0xffffffff
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff


def read_events(stream):
    header_size = struct.calcsize('<QI')
    len_size = struct.calcsize('<Q')
    footer_size = struct.calcsize('<I')

    while True:
        header = stream.read(header_size)
        if len(header) == 0:
            break
        elif len(header) < header_size:
            raise SummaryReaderException('unexpected EOF (expected a %d-byte header, got %d bytes)' %
                                         (header_size, len(header)))
        data_len, len_crc = struct.unpack('<QI', header)
        print('%d bytes' % data_len)
        len_crc_actual = masked_crc(header[:len_size])
        if len_crc_actual != len_crc:
           raise SummaryReaderException('incorrect length CRC (%d != %d)' %
                                        (len_crc_actual, len_crc))

        data = stream.read(data_len)
        if len(data) < data_len:
            raise SummaryReaderException('unexpected EOF (expected %d bytes, got %d)' %
                                         (data_len, len(data)))
        yield Event.FromString(data)

        footer = stream.read(footer_size)
        if len(footer) < footer_size:
            raise SummaryReaderException('unexpected EOF (expected a %d-byte footer, got %d bytes)' %
                                         (footer_size, len(footer)))
        data_crc, = struct.unpack('<I', footer)
        data_crc_actual = masked_crc(data)
        if data_crc_actual != data_crc:
           raise SummaryReaderException('incorrect data CRC (%d != %d)' %
                                        (data_crc_actual, data_crc))


def write_events(stream, events):
    for event in events:
        data = event.SerializeToString()
        len_field = struct.pack('<Q', len(data))
        len_crc = struct.pack('<I', masked_crc(len_field))
        data_crc = struct.pack('<I', masked_crc(data))
        stream.write(len_field)
        stream.write(len_crc)
        stream.write(data)
        stream.write(data_crc)
