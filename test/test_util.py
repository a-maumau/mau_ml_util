# print in color
class str_style:
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    INVISIBLE = '\033[8m'
    STRIKETHROUGH = '\033[9m'
    END = '\033[0m'

class str_fg:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    DEFAULT = '\033[39m'
    END = '\033[0m'

class str_bg:
    BLACK = '\033[40m'
    RED = '\033[41m'
    GREEN = '\033[42m'
    YELLOW = '\033[43m'
    BLUE = '\033[44m'
    PURPLE = '\033[45m'
    CYAN = '\033[46m'
    WHITE = '\033[47m'
    DEFAULT = '\033[49m'
    END = '\033[0m' 

def print_test_starting(module_name=""):
    print("module test: {}".format(module_name))
    print("-------------------------\n")

msg_test_passed = "\n------ test {}{}passed{}{} ------\n".format(str_bg.GREEN, str_fg.BLACK, str_fg.END, str_bg.END)
def print_test_passed():
    print(msg_test_passed)

msg_test_failed = "\n------ test {}{}failed{}{} ------\n".format(str_bg.RED, str_fg.WHITE, str_fg.END, str_bg.END)
def print_test_failed():
    print(msg_test_failed)

# in case when I make a new test printing style.
def print_test_log(msg):
    print(msg)

def return_test_passed():
    return True

def return_test_failed():
    return False
 