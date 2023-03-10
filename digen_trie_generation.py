import dill as pickle
import sys
import pandas as pd
from tpot import TPOTClassifier
from digen import Benchmark
from os.path import exists
from exploration_trie import PipelineTrie, extract_labels
import exploration_trie as et

benchmark=Benchmark()
directoryevs = ["lex_final"]
result = {}
diversity_scores= {}
for j in [2,4,7,14,23,24,25,27,28,30,32,35,40]:

    print(j)
    diversity_scores[j] = {}
    dataset=benchmark.load_dataset('digen'+str(j))

    X, Y = extract_labels(dataset, "target")
    ev = []

    tpot = TPOTClassifier(verbosity=2, population_size=1, generations=1)
    tpot.fit(X, Y)

    for directoryev in directoryevs:
        temp_ev = []
        #pipeline_trie = PipelineTrie()
        diversity_scores[j][directoryev] = []
        for i in range(40):
            pipeline_trie = PipelineTrie()
            # if not exists(f"/Users/matsumoton/Library/CloudStorage/Box-Box/tpot_benchmark_data/results_pop40_gen20_{directoryev}/pipelines/digen{j}_run_{i}_evaluated_individuals.pkl") :
            if not exists(f"./results/digen/results_pop40_gen20_{directoryev}/pipelines/digen{j}_run_{i}_evaluated_individuals.pkl") :
            
                continue
            with open(f"./results/digen/results_pop40_gen20_{directoryev}/pipelines/digen{j}_run_{i}_evaluated_individuals.pkl", 'rb') as file:
                unpickler = pickle.Unpickler(file)
                result = unpickler.load()
                for k , v in result.items():
                    #print(k)
                    pipeline_trie.insert(k,v,tpot._pset)
            pipeline_trie.display(f"./results/digen/results_pop40_gen20_{directoryev}/trie_network/digen{j}_run{i}_ds_{pipeline_trie.root.diversity_score}")
            #if i in [5]:
                #pipeline_trie.display(f"{directoryev}_digen{j}_run{i}_ds_{pipeline_trie.root.diversity_score}")
                
            print(str(j) + ' ' +directoryev+' '+str(i) + ' ' + str([pipeline_trie.root.diversity_score,pipeline_trie.root.max_score]))
            diversity_scores[j][directoryev].append([pipeline_trie.root.diversity_score,pipeline_trie.root.max_score])
            
    
