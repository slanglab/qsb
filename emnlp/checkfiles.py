from __future__ import print_function
import os,sys

def check_file_git_status(filename):
    cmd = "git status --ignored -s {}".format(filename)
    out = os.popen(cmd).read().strip()
    if not out and os.path.exists(filename):
        return ":) {:<15} \t On filesystem & git".format(filename)
    elif not out:
        return ":O {:<15} \t Doesn't exist on filesystem".format(filename)
    else:
        tag = out.split()[0]
        rest = " ".join(out.split()[1:])
        descr = {"M":"Modified on filesystem", "!!": "On filesystem, but not on git"}.get(tag,"")
        return "{:>2} {:<15} \t {}".format(tag,rest,descr)

# filenames = sys.argv[1:]

grep_cmd = r"""  grep -Po '\\include.*' *.tex | perl -pe 's/.*\{//; s/\}.*//' """
filenames = [L.strip() for L in os.popen(grep_cmd)]

for f in filenames:
    print(check_file_git_status(f))


