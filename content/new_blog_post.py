#!/usr/bin/env python3
import os

import click


def main():
    # get directory name from user
    postname = click.prompt("Enter blog post name", type=str).strip()
    if " " in postname:
        print("Cannot have spaces in directory names")
        return

    # make directory and put template in it
    os.system(f"mkdir blog/{postname}")
    os.system(f"cp ~/projects/scripts/templates/hello-world/index.md ./blog/{postname}")
    indexpage = f"./blog/{postname}/index.md"

    # TODO: change date to current time
    # content = ""
    # with open(indexpage, "r") as f:
    # content = f.read()
    os.system(f"nvim {indexpage}")


if __name__ == "__main__":
    main()
