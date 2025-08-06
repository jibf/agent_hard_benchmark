import os


def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
        return content


prefix = os.path.abspath(".")

Backend_prompt = {
    "1": open_file(prefix + "/prompts/add_table.txt"),
    "2": open_file(prefix + "/prompts/revise_table.txt"),
    "3": open_file(prefix + "/prompts/map_table.txt"),
    "4": open_file(prefix + "/prompts/refresh_table.txt"),
    "5": open_file(prefix + "/prompts/add_text.txt"),
    "6": open_file(prefix + "/prompts/revise_text.txt"),
    "7": open_file(prefix + "/prompts/map_text.txt"),
    "8": open_file(prefix + "/prompts/refresh_text.txt"),
    "9": open_file(prefix + "/prompts/add_vector.txt"),
    "10": open_file(prefix + "/prompts/delete_vector.txt"),
    "11": open_file(prefix + "/prompts/map_vector.txt"),
    "12": open_file(prefix + "/prompts/refresh_vector.txt"),
}
