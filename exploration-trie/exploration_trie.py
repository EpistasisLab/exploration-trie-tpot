
from cmath import inf


import matplotlib.pyplot as plt
import numpy as np

import deap
from deap import creator


import pydot 
from IPython.display import Image, display

import networkx as nx
import random
import math

class TrieNode:
 
    def __init__(self, primitive):
        self.primitive = primitive
        self.path = 'root'
        self.traverse_count = 0
        self.total_cv_score = []
        self.generation = []
        self.children = {}
        self.parents = []
        self.depth = 0
        self.max_score = -inf
        self.min_score = inf
        self.diversity_score = 0
 
class PipelineTrie(object):
 
    def __init__(self):
        self.root = TrieNode("")
        
    def insert(self, pipeline_str,pipeline_data,pset):

        def prim_to_list(prim, args):
            if isinstance(prim, deap.gp.Terminal):
                return None
            return [prim.name] + args
        def remove_none(obj):
            if isinstance(obj, (list, tuple, set)):
                return type(obj)(remove_none(x) for x in obj if x is not None)
            elif isinstance(obj, dict):
                return type(obj)((remove_none(k), remove_none(v))
                for k, v in obj.items() if k is not None and v is not None)
            else:
                return obj

        pipeline = creator.Individual.from_string(pipeline_str, pset)

        #convert pipeline into a list and change all hyperparameters to None
        tree = []
        stack = []
        for node in pipeline:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                tree = prim_to_list(prim, args)
                if len(stack) == 0:
                    break  # If stack is empty, all nodes should have been seen
                
                stack[-1][1].append(tree)
        
        #remove all Nones
        tree = remove_none(tree)
        
        #dfs through the tree and integrate into trie
        stack = []
        stack.append(tree)
        trie_stack = [self.root]

        while stack:
            s = stack.pop()
            node = trie_stack.pop()
            cur_depth = node.depth+1
            
            if (s[0]) not in node.children:
                node.children[(s[0])] = TrieNode(s[0])
                node.children[(s[0])].parents = np.append(node.parents,node)
                #add a value to the root diversity metric
                #self.root.diversity_score =  self.root.diversity_score + 1/cur_depth**2
                temp_depth = 1
                for tempnode in node.parents:
                    tempnode.diversity_score =  tempnode.diversity_score + 1/temp_depth**2
                    temp_depth = temp_depth + 1
            node.children[(s[0])].traverse_count = node.children[(s[0])].traverse_count + 1
            node.children[(s[0])].total_cv_score.append(pipeline_data["internal_cv_score"])
            node.children[(s[0])].generation.append(pipeline_data["generation"])
            node.children[(s[0])].depth = cur_depth
            if not math.isnan(pipeline_data["internal_cv_score"]) and not math.isinf(pipeline_data["internal_cv_score"]):
                node.children[(s[0])].min_score = min(node.children[(s[0])].min_score,pipeline_data["internal_cv_score"])
                node.children[(s[0])].max_score = max(node.children[(s[0])].max_score,pipeline_data["internal_cv_score"])
                self.root.min_score = min(self.root.min_score,pipeline_data["internal_cv_score"])
                self.root.max_score = max(self.root.max_score,pipeline_data["internal_cv_score"])
            if node.path != 'root':
                node.children[(s[0])].path = node.path + '-' + s[0]
            else:
                node.children[(s[0])].path = s[0]
            if len(s[1:]) > 0:
                stack.extend(s[1:])
                for i in range(len(s[1:])):
                    trie_stack.append(node.children[(s[0])])
                    

                
    def display(self,filename, depth=100):
        import networkx as nx
        from pyvis.network import Network
        import matplotlib as mpl

        def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
            if mix < 0:
                mix = 0
            if mix > 1:
                mix = 1
            c1=np.array(mpl.colors.to_rgb(c1))
            c2=np.array(mpl.colors.to_rgb(c2))
            return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

        c1='red' #blue
        c2='green' #green

        graph = pydot.Dot(graph_type='graph') 
        stack = [self.root]
        parent_stack = []

        max_height = depth
        while stack:
            s = stack.pop()
            if s.depth >= max_height:
                continue
            for k in s.children.keys():
                stack.append(s.children[k])
                temp =  [v for v in s.total_cv_score if not math.isnan(v) and not math.isinf(v)]
                if len(temp) :
                    parentnodeaccuracy =(sum(temp)/len(temp))
                    if parentnodeaccuracy > self.root.max_score:
                        parentnodeaccuracy = self.root.max_score
                    parentnodecolor = colorFader(c1,c2,(parentnodeaccuracy-self.root.min_score)/(self.root.max_score-self.root.min_score))
                else:
                    parentnodeaccuracy = 'NA'
                    parentnodecolor = "#666666"
                    
                temp =  [v for v in s.children[k].total_cv_score if not math.isnan(v) and not math.isinf(v)]
                if len(temp) :
                    childaccuracy = (sum(temp)/len(temp))
                    #floating point 0.00...01 issue
                    if childaccuracy > self.root.max_score:
                        childaccuracy = self.root.max_score
                    childcolor = colorFader(c1,c2,(childaccuracy-self.root.min_score)/(self.root.max_score-self.root.min_score))
                    
                else:
                    childaccuracy = 'NA'
                    childcolor = "#666666"
                
                graph.add_node(pydot.Node(s.path,label=s.primitive+'\n'+str(parentnodeaccuracy),color=parentnodecolor,size=10*(math.tanh(-s.depth+4)+2)))
                graph.add_node(pydot.Node(s.children[k].path,label=s.children[k].primitive+'\n'+str(childaccuracy),color=childcolor,size=10*(math.tanh(-s.children[k].depth+4)+2)))
                
                edge = pydot.Edge(s.path, s.children[k].path,weight=1,color='#515ba3',value=math.log(s.children[k].traverse_count))
                graph.add_edge(edge)



        G = nx.nx_pydot.from_pydot(graph)
        nt = Network(bgcolor='#333333', font_color='white', height="100%",width="100%")
        nt.from_nx(G)
        nt.show(filename+'.html')
 



def extract_labels(df, labelname):
    y = df[labelname].copy(deep=True)
    x = df.drop(labelname, axis=1)
    x, y = shuffle(x, y)
    x = x.to_numpy()
    y = y.to_numpy()
    return x, y

