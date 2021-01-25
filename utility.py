import sys
import time


def print_status(string):
    sys.stdout.write(f'\r{string}')
    sys.stdout.flush()


def clear_status():
    sys.stdout.write('\r')
    sys.stdout.flush()


class ProcessTimeStopwatch:
    def __init__(self):
        self._timestamp = None

    def start(self):
        self._timestamp = time.process_time()

    def stop(self):
        elapsed = time.process_time() - self._timestamp

        return elapsed


class ProcessRunElapsed(ProcessTimeStopwatch):
    def __init__(self, callback):
        super(ProcessRunElapsed, self).__init__()
        self.callback = callback

    def __call__(self, *args, **kwargs):
        super().start()
        res = self.callback(*args, **kwargs)
        elapsed = super().stop()

        if res:
            return elapsed, res
        return elapsed


def stringify_arguments(prefix, string_delimiter=', ', **kwargs):
    items = [prefix]

    for k, v in kwargs.items():
        items.append(''.join([k, ': ', v]))
        items.append(string_delimiter)

    return ''.join(items[:-1])


# ref: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch14s08.html
def get_debug_information():
    debug_ctx = sys._getframe(1)

    line_number = debug_ctx.f_lineno
    filename = debug_ctx.f_code.co_filename
    function_name = debug_ctx.f_code.co_name

    return f'{function_name} ({filename}, {line_number})'
