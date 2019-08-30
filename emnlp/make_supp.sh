#!/bin/bash
set -eux
rm -rf supp_appendix supp_appendix.zip
mkdir supp_appendix
cp appendix.pdf supp_appendix 
zip -r supp_appendix supp_appendix

