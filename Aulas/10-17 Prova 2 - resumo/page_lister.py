#!/usr/bin/env python3.7


"""
"""


import inspect
import os
import PyPDF2
import re
import sys


def main(pdfpath):
    """"""

    pdffile = PyPDF2.PdfFileReader(open(pdfpath, 'rb'))
    pagecount = pdffile.getNumPages()

    for i in range(1, pagecount+1):
        template = f"""
        {i}
        \\noindent\\includegraphics[page={i},width=\\columnwidth,keepaspectratio]{{{pdfpath}}}
        """
        template = inspect.cleandoc(template)

        print(template)


if __name__ == '__main__':
    main(sys.argv[1])
