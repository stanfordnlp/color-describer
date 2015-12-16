import argparse
import os
import sys
import json
import logfile
import traceback
import StringIO
import contextlib
import __builtin__


_options_parser = argparse.ArgumentParser(conflict_handler='resolve')
_options_parser.add_argument('--run_dir', '-R', type=str, default=None)

def get_options_parser():
    return _options_parser


_options = None

def options(allow_partial=False):
    global _options

    if allow_partial:
        opts, extras = _options_parser.parse_known_args()
        return opts

    if _options is None:
        _options = _options_parser.parse_args()
        dump_pretty(vars(_options), 'config.json')
    return _options


def get_file_path(filename):
    opts = options(allow_partial=True)
    if not opts.run_dir:
        return None
    return os.path.join(opts.run_dir, filename)


def open(filename, *args, **kwargs):
    file_path = get_file_path(filename)
    if not file_path:
        # create a dummy file because we don't have a run dir
        return contextlib.closing(StringIO.StringIO())
    return __builtin__.open(file_path, *args, **kwargs)


def boolean(arg):
    """Convert a string to a bool treating 'false' and 'no' as False."""
    if arg in ('true', 'True', 'yes', '1', 1):
        return True
    elif arg in ('false', 'False', 'no', '0', 0):
        return False
    else:
        raise argparse.ArgumentTypeError(
            'could not interpret "%s" as true or false' % (arg,))


def redirect_output():
    outfile = get_file_path('stdout.log')
    if outfile is None: return
    logfile.log_stdout_to(outfile)
    logfile.log_stderr_to(get_file_path('stderr.log'))

redirect_output()


def dump(data, filename, lines=False, *args, **kwargs):
    try:
        with open(filename, 'w') as outfile:
            if lines:
                for item in data:
                    json.dump(item, outfile, *args, **kwargs)
                    outfile.write('\n')
            else:
                json.dump(data, outfile, *args, **kwargs)
    except IOError:
        traceback.print_exc()
        print >>sys.stderr, 'Unable to write %s' % filename
    except TypeError:
        traceback.print_exc()
        print >>sys.stderr, 'Unable to write %s' % filename


def dump_pretty(data, filename):
    dump(data, filename,
         sort_keys=True, indent=2, separators=(',', ': '))