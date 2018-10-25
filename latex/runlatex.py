#!/usr/bin/env python
"""
Latex command-line shortcuts.  Symlink this script as several commands:
  - gotex [target]:     Full latex build: pdflatex;pdflatex;bibtex;pdflatex
  - qtex [target]:      Quick latex build: pdflatex
  - cleantex [target]:  Remove intermediate files from a previous build.

This script adds a few capabilities compared to running the pdflatex/bibtex
commands yourself:
  - Output formatting/cleaning.  Try to remove noninformative output.
  - Suppress output on certain successful commands.
  - Automatically try to figure out what's the "main" latex file to build if
    you don't supply a target.
  - Run in non-interactive mode by default. If you have weird errors you need
    to see or respond to, override with -i.

Website: https://github.com/brendano/runlatex
"""
# todo: look into what ST LatexTools or other editor/ide integrations do

import sys,os,re,glob

def texref():
    """show references.. weirdly specific to certain conventions.."""
    basecmd = r"""grep -Po 'label\{PREFIX:\S+?\}' *.tex|sort|uniq  |
     perl -pe 's/label//g' |
     perl -pe 's/:/\t/' | tsv2fmt"""
    os.system(basecmd.replace("PREFIX","c"))
    print
    os.system(basecmd.replace("PREFIX","s"))
    print
    os.system(basecmd.replace("PREFIX","f"))
    
def findtarget():
    if FLAGS.target:
        target = FLAGS.target[0]
        target = re.sub(r'\.tex$','', target)
        target = re.sub(r'\.pdf$','', target)
        return target

    ## find on disk
    files = list(glob.glob("*.tex"))
    files = [f for f in files if r'\documentclass' in open(f).read(10000)]
    if not files:
        myprint("*** Couldn't find a file to work on")
        usage()

    # If multiple "main" files: break ties via most-worked and recently
    # modified
    if os.system("git log 2>&1 | head >/dev/null") == 0:
        ## Prefer the most-worked-on file
        def numcommits(f):
            p = os.popen("git log --oneline {f}".format(f=f))
            lines = [L for L in p.readlines() if L.strip()]
            return len(lines)
    else:
        def numcommits(f): return -1
    files.sort(key=lambda f: (-numcommits(f), -os.stat(f).st_mtime, f))
    target = files[0]
    target = re.sub(r'\.tex$','', target)
    myprint("Selected target: %s" % target)
    return target
        
def do_open(target):
    # should try to be smarter about whether the file is already open
    # but i couldnt figure out how to do that
    # os.system("open -a Preview")
    os.system("open {target}.pdf".format(**locals()))

def run_pdflatex_log(target, logfile, stepnum=None, **clean_args):
    # run pdflatex with logfile
    cmd1 = "pdflatex"
    if FLAGS.ignore_errors:
        pass
    else:
        cmd1 += " -halt-on-error"
    if FLAGS.interactive:
        # no logging I guess
        cmd1 = "{cmd1} {target} | tee {logfile}".format(**locals())
    else:
        cmd1 = "{cmd1} -interaction=nonstopmode {target} > {logfile}".format(**locals())

    cmd = """
    {setflags}
    mkdir -p texlogs
    export max_print_line=10000
    export error_line=10000
    {cmd1}
    """.format(
            setflags="" if FLAGS.ignore_errors else "set -eu -o pipefail",
            **locals())

    out = cmd1.strip().split("\n")[-1].strip()
    if stepnum is not None:
        out = "[%s] %s" % (stepnum, out)
    myprint(out)
    ret = os.system(cmd)
    if ret !=0: 
        if FLAGS.interactive:
            myprint("Command failed.")
        elif FLAGS.verbose:
            myprint("Command failed. Output:")
            print open(logfile).read()
            myprint("Above: raw output after failed command")
            myprint("Full output: {logfile}".format(**locals()))
        else:
            myprint("Command failed. Cleaned output (full output in {logfile}):\n".format(**locals()))
            print clean_output(open(logfile).read(), **clean_args)
            myprint("Above: cleaned output after failed command: {cmd1}".format(**locals()))
            myprint("Full output: {logfile}".format(**locals()))
    return ret

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def myprint(s):
    print bcolors.OKBLUE + "[runlatex] " + s + bcolors.ENDC

def qtex():
    target = findtarget()
    logfile = "texlogs/%s.qtex.log" % target
    ret = run_pdflatex_log(target, logfile)
    if ret==0:
        if FLAGS.interactive:
            pass
        elif FLAGS.verbose:
            print open(logfile).read()
        else:
            myprint("Cleaned output:")
            print clean_output(open(logfile).read())
        myprint("Successfully built {target}.pdf".format(**locals()))
    else:
        print clean_output(open(logfile).read())
        myprint("Error {ret} while compiling {target}.pdf".format(**locals()))


def gotex():
    target = findtarget()
    os.system("rm -f texlogs/%s.gotex.[1-4].log" % target)
    CE = not FLAGS.ignore_errors # "care about errors"
    def gopipeline():
        ret = run_pdflatex_log(target, "texlogs/{target}.gotex.1.log".format(**locals()), stepnum=1, clean_natbib=True)
        if ret != 0 and CE: return ret
        cmd = "bibtex {target} | tee texlogs/{target}.gotex.2.log".format(**locals())
        myprint("[2] %s" % cmd)
        ret = os.system(cmd)
        if ret != 0 and CE: return ret
        ret = run_pdflatex_log(target, "texlogs/{target}.gotex.3.log".format(**locals()),stepnum=3)
        if ret != 0 and CE: return ret
        ret = run_pdflatex_log(target, "texlogs/{target}.gotex.4.log".format(**locals()),stepnum=4)
        return ret
    ret = gopipeline()
    if ret==0:
        files = glob.glob("texlogs/%s.gotex.*.log" % target)
        logfile = sorted(files)[-1]
        print clean_output(open(logfile).read())
        myprint("Successfully built {target}.pdf".format(**locals()))
    else:
        myprint("Error {ret} while compiling {target}.pdf".format(**locals()))

def clean_output(s, clean_natbib=False):
    lines = s.split("\n")
    regexes = [
    '^This is pdfTeX',
    r'^ *restricted *\S+ *enabled.{0,5}$',
    r'^entering extended mode *$',
    '^LaTeX2e',
    '^Babel .*loaded.{0,5}$',
    '^Document Class:',
    '^Conference Style for',
    'ABD: EveryShipout',
    'For additional information on amsmath',
    'Package hyperref Message: Driver',
    'Loading MPS to PDF converter',
    '^Transcript written on',
    '^\*geometry\*',
    '^\*+$',
    '^\* *Local config',
    '[pP]ackage.*algorithm2e.*[rR]elease',
    'algorithm2e.*mailing list',
    'subscribe by emailing',
    'Author: Christophe Fiorio',
    '^ *(\[\])+ *$',
# LaTeX Font Warning: Font shape `OMS/pplx/m/n' undefined
# (Font)              using `OMS/cmsy/m/n' instead
# (Font)              for symbol `textbullet' on input line 9.
    'LaTeX Font Warning',
    r'\(Font\)',
            ]
    if clean_natbib:
        # Package natbib Warning: Citation `Chen2014NN' on page A-8 undefined on input line 412.
        regexes.append('Package natbib Warning.*Citation.*undefined')
    regex = '|'.join('('+x+')' for x in regexes)
    lines = [L for L in lines if not re.search(regex, L)]

    lines2=[]
    mode=False
    for L in lines:
        if re.search(r'^(Underfull|Overfull) ', L):
            mode=True
        elif mode and r'\OT1/' in L:
            pass
        else:
            mode=False
        if not mode:
            lines2.append(L)

    # Delete filepaths of loaded modules and extraneous parens about them.
    # BASE = "/usr/local/texlive"
    f = os.popen("which pdflatex").read().strip()
    BASE = re.sub('/bin/.*', "", f)

    s = "\n".join(L.strip() for L in lines2)
    s = re.sub(r'\(\./[^/]*\.(tex|aux|out|sty|bbl)\)?', "", s)
    s = re.sub(r'\(+%s\S*' % BASE, "", s)
    s = re.sub(r'\[.{0,5}\{%s\S*?\}\]' % BASE, "", s)
    s = re.sub(r'\{%s\S*?\}' % BASE, "", s)
    s = re.sub(r'\<%s\S*?\>' % BASE, "", s)
    for i in xrange(5):
        s = re.sub(r'(^|\s)[\(\)](\s|$)', "", s)
    s = "\n".join(L.strip() for L in s.split("\n") if L.strip())
    return s

def cleantex(target=None, skip_global=False):
    """cleantex [TARGET]:   Eliminate .out crap around a .tex file.
                     Without TARGET, clean all *.tex files in current 
                     directory.
    """
    if not skip_global:
        cmd = "rm -rf texlogs texput.log .DS_Store"
        if FLAGS.verbose: myprint(cmd)
        os.system(cmd)

    if target is None:
        texfiles = glob.glob("*.tex")
        if not texfiles:
            myprint("No texfiles here.")
            return
        myprint("Cleaning crap around files: %s" % repr(texfiles))
        for texfile in texfiles:
            cleantex(texfile, skip_global=True)
        return

    assert target.strip()

    target = re.sub(r'\.tex$','', target)
    target = re.sub(r'\.pdf$','', target)
    cmd = """
    rm -f %s.{aux,bbl,blg,log,dvi,out,nav,snm,toc,synctex.gz*,fdb_latexmk,fls}
    """ % (target,)
    if FLAGS.verbose: myprint(cmd.strip())
    os.system(cmd)

def usage():
    pp.print_help()
    sys.exit(1)


import argparse
pp = argparse.ArgumentParser(description=__doc__.strip(), 
        formatter_class=argparse.RawDescriptionHelpFormatter)
pp.add_argument('target', nargs='*')
pp.add_argument('-v', '--verbose', action='store_true', help="Show all output, instead of trying to clean/hide it.")
pp.add_argument('-i', '--interactive', action='store_true', help="Run in interactive mode with all output -- this is the default way of running the pdflatex command.")
pp.add_argument('-e', '--ignore-errors', action='store_true')

# Find the desired subcommand from symlink name (ideally)
cmd = None
args = sys.argv
while True:
    if len(args)==0:
        usage()
    front = args.pop(0)
    if os.path.basename(front).startswith('python'):
        continue
    elif os.path.basename(front)=='runlatex.py':
        if len(args)==0:
            cmd = 'gotex'
            break
        else:
            continue
    else:
        cmd = os.path.basename(front)
        break

FLAGS = pp.parse_args(args=args)
# print FLAGS; sys.exit()

if cmd not in dir():
    usage()

try:
    eval(cmd)()
except TypeError:
    raise
    usage()


