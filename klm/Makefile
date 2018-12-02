
lang_model:
	python alls_maker.py > vox.txt
	./kenlm/build/bin/lmplz -o 3 -S 80% -T /home/ahandler/treeops/klm/tmp/ < ../gigaword_lm/all.eng.gigaword > gigaword.arpa	
	./kenlm/build/bin/lmplz -o 1 -S 80% -T /home/ahandler/treeops/klm/tmp/ < ../gigaword_lm/all.eng.gigaword > gigaword.arpa.unigram	
	./kenlm/build/bin/build_binary gigaword.arpa gigaword.klm
	#./kenlm/build/bin/build_binary gigaword.arpa.unigram gigaword.klm.unigram # just use the .arpa file I think
