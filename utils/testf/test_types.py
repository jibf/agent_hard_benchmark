import string
import random
from collections import defaultdict

fontlist = [
    "Lucida Console",
    "lucida console",
    "Tahoma",
    "tahoma",
    "georgia",
    "Georgia",
    "Arial",
    "verdana",
    "Verdana",
    "arial",
    "helv",
    "Times New Roman",
    "Times-roman",
    "times new roman",
    "courier",
    "Courier",
    "courier-oblique",
    "Courier-Oblique",
    "courier-bold",
    "Courier-Bold",
    "courier-boldoblique",
    "Courier-BoldOblique",
    "helvetica",
    "Helvetica",
    "helvetica-oblique",
    "Helvetica-Oblique",
    "helvetica-bold",
    "Helvetica-Bold",
    "helvetica-boldoblique",
    "Helvetica-BoldOblique",
    "times-roman",
    "Times-Roman",
    "times-italic",
    "Times-Italic",
    "times-bold",
    "Times-Bold",
    "times-bolditalic",
    "Times-BoldItalic",
    "symbol",
    "Symbol",
    "zapfdingbats",
    "ZapfDingbats",
    "helv",
    "Helvetica",
    "heit",
    "Helvetica-Oblique",
    "hebo",
    "Helvetica-Bold",
    "hebi",
    "Helvetica-BoldOblique",
    "cour",
    "Courier",
    "coit",
    "Courier-Oblique",
    "cobo",
    "Courier-Bold",
    "cobi",
    "Courier-BoldOblique",
    "tiro",
    "Times-Roman",
    "tibo",
    "Times-Bold",
    "tiit",
    "Times-Italic",
    "tibi",
    "Times-BoldItalic",
    "symb",
    "Symbol",
    "zadb",
    "ZapfDingbats",
    "Calibri",
    "calibri",
    "helv-bold",
    "helv-light",
    "Arial Bold",
    "Courier New",
]

TEXT_ALIGN_CENTER = fitz.TEXT_ALIGN_CENTER
TEXT_ALIGN_RIGHT = fitz.TEXT_ALIGN_RIGHT
TEXT_ALIGN_JUSTIFY = fitz.TEXT_ALIGN_JUSTIFY
TEXT_ALIGN_LEFT = fitz.TEXT_ALIGN_LEFT


def generate_random_string(length):
    all_chars = string.ascii_letters + string.digits
    random_string = "".join(random.choice(all_chars) for _ in range(length))
    return random_string


def file_format(filepath: any):
    if filepath in ["missing", "Missing"]:
        validity = True
    elif type(filepath) is str and filepath[-4:] == ".pdf":
        validity = True
    else:
        validity = False
    fileformat = {
        "filepath": (filepath, validity),
    }
    return defaultdict(list, fileformat)


class fileobject:
    def __init__(self, filepath: any = "Unknown"):
        self.fileblock = file_format(filepath)

    def save(self, new_path: any = "Unkown"):
        # default frame
        save_format = {
            "save_path": (
                new_path,
                True if type(new_path) is str and new_path[-4:] == ".pdf" else False,
            ),
        }

        self.fileblock["save"].append(save_format)
        # save actions
        if save_format["save_path"][1]:
            if self.fileblock["change_maker"]:
                for a, _ in enumerate(self.fileblock["change_maker"]):
                    try:
                        self.fileblock["change_maker"][a]["save"] = True
                        for tool in self.fileblock["change_maker"][a]["tool_callings"]:
                            tool["save"] = True
                    except Exception as e:
                        continue
                for a, _ in enumerate(self.fileblock["post_change_maker"]):
                    try:
                        self.fileblock["post_change_maker"][a]["save"] = True
                        for tool in self.fileblock["post_change_maker"][a][
                            "tool_callings"
                        ]:
                            tool["save"] = True
                    except Exception as e:
                        continue
                for a, _ in enumerate(self.fileblock["recorder"]):
                    try:
                        self.fileblock["recorder"][a]["save"] = True
                        for tool in self.fileblock["recorder"][a]["tool_callings"]:
                            tool["save"] = True
                    except Exception as e:
                        continue


class extrracted_anno:
    def __init__(
        self,
        doc: tuple = ("Unknown", False),
        page: tuple = ("Unknown", False),
        order_or_annocolor: tuple = ("Unknown", False),
    ):
        self.doc = doc
        self.page = page
        self.order_or_annocolor = order_or_annocolor
