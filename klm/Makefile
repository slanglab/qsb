
lang_model:
	python alls_maker.py > fa.txt
	./kenlm/kenlm/build/bin/lmplz -o 3 -S 80% -T tmp/ < fa.txt > fa.arpa
	./kenlm/kenlm/build/bin/build_binary fa.arpa fa.klm
	python arpa2unigram_probs.py	
	python query.py
