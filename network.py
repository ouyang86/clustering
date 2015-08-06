#!/usr/bin/python
import random
import numpy as np
from collections import deque
from scipy import linalg
#from scipy.cluster.vq import vq, whiten, kmeans2
from scipy.stats import norm
from scipy import sparse as sps
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
from sklearn import metrics
import igraph as ig
from igraph.drawing.text import TextDrawer
import cairo


#randomly simulate a cluster in a network
#imput adj is a 2D array of adjacency matrix
#input node_list is a list of node index
#p is the wiring probability
def simulate_cluster(adj, node_list, p):
    for i in node_list:
        for j in node_list:
            sim = random.random()
            if i < j and sim < p:
                adj[i,j] = 1
                adj[j,i] = 1
                
 
#randomly simulate edges between clusters
#node_list1 and node_list2 should have nothing in common
#p is the wiring probability
def simulate_link(adj, node_list1, node_list2, p):
    for i in node_list1:
        for j in node_list2:
            sim = random.random()
            if sim < p:
                adj[i,j] = 1
                adj[j,i] = 1


#create adjacency list to adjacency sparse matrix
def adjlist_to_adj(adjlist):
    row = []
    col = []
    for i, item in enumerate(adjlist):
        for j in item:
            row.append(i)
            col.append(j)
    data = np.ones(len(row))
    adj = sps.coo_matrix((data,(row, col)))
    return adj


def temp2(n, p, p0):
    out = 0
    for i in range(n):
        a = np.zeros((45, 45))
        simulate_link(a, range(20), range(20,45), p)
        av = np.sum(a[np.ix_(range(20), range(20, 45))])/float(20*25)
        if av > p0:
            out += 1
    return out / float(n)


def temp(N,n, p, p0):
    out = 0
    for i in range(N):
        prop = 0
        for k in range(n):
            sim = random.random()
            if sim <= p:
                prop += 1
        if prop / float(n) >= p0:
            out += 1
    return out / float(N)
        

#create a function to generate linear operator for 
#clique-based clusering cost matrix
#a is the CSR sparse adjacency matrix
#p is a real number in [0,1]
def clique_fun(a, p):
    def reward(v):
        v_len = len(v)
        if v.shape == (v_len,):
            v = v.reshape(-1, 1)
            v = np.matrix(v)
        if v.shape == (v_len, 1):
            v_ones = np.matrix(np.ones(v_len)).T
            out = a*v - p*v_ones*(v_ones.T*v) + p*v
            return out
    return reward
    
  
#create a function to generate linear operator for
#modularity matrix
#a is the CSR sparse adjacency matrix
#lamb is the multiresolution parameter
def mod_fun(a, deg, total_deg, nul=True, lamb=1.0):
    def reward(v):
        v_len = len(v)
        if v.shape == (v_len,):
            v = v.reshape(-1, 1)
            v = np.matrix(v)
        if v.shape == (v_len, 1):
            v_ones = np.matrix(np.ones(v_len)).T
            out = a*v - lamb*deg*(deg.T*v)/total_deg
            if nul == False:
                return out
            temp = a*v_ones -lamb*deg*(deg.T*v_ones)/total_deg
            temp = np.array(temp).reshape(-1,)
            out -= sps.diags([temp],[0],format='csr')*v
            return out
    return reward
        



#define a class for network clustering
#adj is the adjacency matrix in sparse matrix data structure
class Clustering(object):
    def __init__(self, adj):
        self.adj = sps.csc_matrix(adj)
        self.cluster = []
        self.density = []
        #self.membership = None
        self.size = adj.shape[0]
        self.index = 0
        self.edgelist = None
        self.g = None
        self.layout = None
        
    #member is a 1D numpy array of membership assignment
    #in a clustering
    def mod_score(self, member=None):
        if member == None:
            member = self.cluster
        else:
            member = list_cluster(member)
        score = np.zeros(len(member))
        for i, item in enumerate(member):
            n = len(item)
            degree = self.adj.sum(axis=0).T
            deg = degree[item]
            total_degree = degree.sum()
            clique = self.adj[:,item]
            clique = clique.tocsr()
            clique = clique[item,:]
            clique = mod_fun(clique, deg, total_degree, 
                             nul=False)            
            score[i] = clique(np.ones(n)).sum()
        return score.sum() / total_degree
        

    def mod_dfs(self, method=1, lamb=1.0, maxiter=50000):
        subnet_list = [np.arange(self.size)]
        self.cluster = []
        self.density = []
        self.index = 0
        degree = self.adj.sum(axis=0).T
        total_degree = degree.sum()
        while subnet_list:
            sub = subnet_list.pop()
            n = len(sub)
            if n == 1:
                self.density.append((1,0))
                self.cluster.append((sub))
                continue
            clique = self.adj[:,sub]
            clique = clique.tocsr()
            clique = clique[sub,:]
            link = clique.sum()
            total = float(n*(n-1))
            clique = mod_fun(clique, degree[sub], total_degree, lamb)
            clique = LinearOperator(shape=(n,n), matvec=clique,
                                    dtype='float64')
            w, v = eigsh(A=clique, k=1, which='LA', maxiter=maxiter,tol=1E-4)
            group_ind = v[:,0] > 0
            group1 = sub[group_ind]
            group2 = sub[-group_ind]
            v = 2.0*group_ind.astype(int) - 1
            out = np.dot(v, (clique*v)) / 2.0
            nul = (clique*np.ones(n)).sum()
            adv = out / total_degree
            if adv > 10**(-6):
                self.index += adv
                subnet_list.append(group1)
                subnet_list.append(group2)
            else:
                self.density.append((n, link/total))
                self.cluster.append(sub)


    def auto_dfs(self, error=0.025, maxiter=50000):
        subnet_list = [np.arange(self.size)]
        self.cluster = []
        self.density = []
        self.index = 0
        total_degree = self.adj.sum()
        while subnet_list:
            sub = subnet_list.pop()
            n = len(sub)
            if n == 1:
                self.density.append((1,0))
                self.cluster.append((sub))
                continue
            clique = self.adj[:,sub]
            clique = clique.tocsr()
            clique = clique[sub,:]
            link = clique.sum()
            density = link / float(n*(n-1))
            p = choose_p(n, density, error)
            clique = clique_fun(clique, p)
            clique = LinearOperator(shape=(n,n), matvec=clique, 
                                    dtype='float64')
            w, v = eigsh(A=clique, k=1, maxiter=maxiter,
                         which='LA')
            group_ind = v[:,0] > 0
            group1 = sub[group_ind]
            group2 = sub[-group_ind]    
            v = 2.0*group_ind.astype(int) - 1
            out = np.dot(v, (clique*v))
            nul = (clique*np.ones(n)).sum()
            adv = (out-nul) / total_degree
            if adv > 10**(-6):
                self.index += adv
                subnet_list.append(group1)
                subnet_list.append(group2)
            else:
                self.density.append((n, density))
                self.cluster.append(sub)


    def dfs(self, p, maxiter=50000):  
        subnet_list = [np.arange(self.size)]
        self.cluster = []
        self.density = []
        self.index = 0
        total_degree = self.adj.sum()
        while subnet_list:
            sub = subnet_list.pop()
            n = len(sub)
            if n == 1:
                self.density.append((1,0))
                self.cluster.append((sub))
                continue
            clique = self.adj.tocsc()[:,sub]
            clique = clique.tocsr()
            clique = clique[sub,:]
            link = clique.sum()
            density = link / float(n*(n-1))
            clique = clique_fun(clique, p)
            clique = LinearOperator(shape=(n,n), matvec=clique, 
                                    dtype='float64')
            w, v = eigsh(A=clique, k=1, maxiter=maxiter,
                         which='LA')
            group_ind = v[:,0] > 0
            group1 = sub[group_ind]
            group2 = sub[-group_ind]    
            v = 2.0*group_ind.astype(int) - 1
            out = np.dot(v, (clique*v))
            nul = (clique*np.ones(n)).sum()
            adv = (out-nul) / total_degree
            if adv > 10**(-6) or nul < 0:
                if len(group1) == len(sub) or len(group2) == len(sub):
                    group1 = random.sample(sub, (n/2))
                    group2 = list(set(sub).difference(set(g1)))
                self.index += adv
                subnet_list.append(group1)
                subnet_list.append(group2)
            else:
                self.density.append((n, density))
                self.cluster.append(sub)
   
    #get membership array from clustering results
    def get_membership(self):
        membership = np.zeros(self.size, dtype=int)
        for i, cluster in enumerate(self.cluster):
            for node in cluster:
                membership[node] = i
        return membership
                
    
    #get edgelist from csc sparse adjacency matrix
    def get_edgelist(self, directed=False):
        if self.adj.nnz == 0:
            return None
        edgelist = []
        cur_col = 0
        for ind, item in enumerate(self.adj.indices):
            if ind >= self.adj.indptr[cur_col+1]:
                cur_col += 1
            if directed or item <= cur_col:
                edgelist.append((item, cur_col))
        return edgelist
                
    #make a plot for the graph
    """
    If membership is None, the self.cluster will be used for
    membership information. Otherwise, membership is used
    If infile is None, plot will not be written to hard drive
    We can specify the path
    
    """
    def plot_network(self, title='Network', membership=None, 
                     outfile=None, width=900, height=600):
        if self.edgelist is None:
            self.edgelist = self.get_edgelist()
        if self.g is None:
            self.g = ig.Graph(self.edgelist)
        if membership is None:
            membership = self.get_membership()
            pal_size = max(1, len(self.cluster))   
        else:
            temp = set()
            for item in membership:
                temp.add(item)
            pal_size = max(1, len(temp))
        pal = ig.ClusterColoringPalette(pal_size)
        visual_style = {}
        visual_style['vertex_color'] = [pal.get(id) \
                                        for id in membership]
        if self.size <= 100:
            visual_style['vertex_size'] = 20
            visual_style['vertex_label'] = [str(id+1) \
                                            for id in xrange(self.size)]
        else:
            visual_style['vertex_size'] = 8
        if not self.layout:
            if self.size <= 200:
                self.layout = self.g.layout('kk')
            else:
                self.layout = self.g.layout('lgl') 
        visual_style['margin'] = (20,50,20,20)
        visual_style['layout'] = self.layout
        visual_style['bbox'] = (width, height)
        figure = ig.plot(self.g, **visual_style)
        figure.redraw()
        ctx = cairo.Context(figure.surface)
        ctx.set_font_size(20)
        drawer = TextDrawer(ctx, title, halign=TextDrawer.CENTER)
        drawer.draw_at(3,30, width=width)
        if outfile:
            figure.save(outfile)
        figure.show()
        
                    
 

#test the probability to split a ER random network
#type 1 error
def test_error1(size, prob, p, n, risk=0.025, lamb=1.0):
    out = np.zeros(n)
    #if p == 'auto':
    #    p = choose_p(size, prob, risk)
    for i in range(n):
        a = np.zeros((size, size))
        simulate_cluster(a, range(size), prob)
        if p == 'auto':
            density = np.sum(a) / (size*(size-1))
            p = choose_p(size, density, risk)
        a = sp.csr_matrix(a)
        net = Clustering(a)
        if p == 'modularity':
            g1, g2, o, nul, w = net.bi_modularity(np.arange(size), lamb)
        else: 
            g1, g2, o, nul, w = net.bi_partition(np.arange(size), p)
        out[i] = min(len(g1), len(g2))/float(size)
    return out.mean(), out.std()/np.sqrt(n)


#test the probability to merge a random network with 
#two clusters
#type 2 error
def test_error2(size1, prob1, size2, prob2, prob, p, n):
    out = 0
    for i in range(n):
        size = size1 + size2
        a = np.zeros((size, size))
        simulate_cluster(a, range(size1), prob1)
        simulate_cluster(a, range(size1, size), prob2)
        simulate_link(a, range(size1), range(size1, size), prob)
        net = Clustering(a)
        g1, g2, o, nul = net.bi_partition(np.arange(size), p)
        c1 = set(range(size1)).difference(set(g1))
        c2 = set(range(size1, size)).difference(set(g2))
        error = min((len(c1)+len(c2)), (size-len(c1)-len(c2)))
        adv = (o-nul) / float(size*(size-1))
        out1 = np.sum(a[np.ix_(g1,g2)]/float(len(g1)*len(g2)))
        out2 = np.sum(a[np.ix_(range(size1), range(size1, size))]/
                      float(size1*size2))
        out += error / float(size)
        stad = np.zeros((size,1))
        stad[:size1,0] = 1.0
        stad[size1:,0] = -1.0
        adj = a - p + p*np.eye(size)
        stad_out = np.dot(stad.T, np.dot(adj,stad))
        print o, stad_out, nul, out1, out2, error/float(size)
    return out / float(n)
    
    
#simulate a network with multiple clusters
#size ia a 1D vector of cluster size
#prob is a 2D array of probability of link within 
#and between clusters
def simulate(size, prob):
    total = np.sum(size)
    k = len(size)
    a = sps.diags([[0]*total],[0], format='lil')
    #a = np.zeros((total, total))
    index = []
    current = 0
    for i in range(k):
        follow = current + size[i]
        index.append(range(current, follow))
        simulate_cluster(a, index[i], prob[i,i])
        current = follow
    for i in range(k):
        for j in range(i):
            simulate_link(a, index[i], index[j], prob[i,j])
    #a = a.tocsr()
    #a = sps.csr_matrix(a)
    return a, index


#get cluster from stochastic block model
def get_sbm_cluster(size):
    k = len(size)
    cluster = []
    current = 0
    for i in range(k):
        follow = current + size[i]
        cluster.append(np.arange(current, follow))
        current = follow
    return cluster

#create dictionary of clustering results
#from a list of clusters
def dict_cluster(cluster):
    out = {}
    for i, item in enumerate(cluster):
        for node in item:
            out[node] = i
    return out


#create a membership array of clustering results
#from a list of clusters
def label_cluster(cluster, net_size=None):
    if not net_size:
        net_size = 0
        for item in cluster:
            net_size += item.shape[0]
    out = np.zeros(net_size, dtype=int)
    for i, item in enumerate(cluster):
        for node in item:
            out[node] = i
    return out


#create a label list from a membership array
#the inverse function of label_cluster
def list_cluster(label):
    out = dict()
    for i, item in enumerate(label):
        out[item] = out.get(item, [])
        out[item].append(i)
    return out.values()


#compute the Normalized MUtual Information
#from clustering
#size is a 1D array of cluster size
#prob is the probability matrix in stochastic model
#p is the parameter p used
#n is the number of simulation
def nmi_testing(size, prob, p, n, error=0.025, method='mine', 
                lamb=1.0, v=0):
    nmi = np.zeros(n)
    for i in xrange(n):
        a, index = simulate(size, prob)
        q = a.shape[0]
        net = Clustering(a)
        if p == 'auto':
            net.auto_dfs(error=error)
        elif p == 'modularity':
            net.mod_dfs(lamb=lamb)
        else:
            net.dfs(p)
        if method.lower() == 'mine':
            nmi[i] = my_nmi(index, net.cluster, version=v)
            continue
        result = label_cluster(q, net.cluster)
        answer = label_cluster(q, index)
        if method.lower() == 'normalized':
            nmi[i] = metrics.normalized_mutual_info_score(answer, result)
        elif method.lower() == 'adjusted':
            nmi[i] = metrics.adjusted_mutual_info_score(answer, result)
    return nmi.mean(), nmi.std()/np.sqrt(n)


#testing clustering results in simulated network
#with stochastic block model
def testing(size, prob, p):
    #network_size = np.sum(size)
    a, index = simulate(size, prob)
    n = a.shape[0]
    net = Clustering(a)
    if p == 'auto':
        net.auto_dfs()
    elif p == 'modularity':
        net.mod_dfs()
    else:
        net.dfs(p)
    result = dict_cluster(net.cluster)
    answer = dict_cluster(index) 
    out = 0
    error = np.zeros(prob.shape)
    count = np.zeros(prob.shape)
    for i in range(n):
        for j in range(i):
            r = result[i] == result[j]
            c = answer[i] == answer[j]
            if r != c:
                out += 1.0
                error[answer[i],answer[j]] += 1.0
                if not c:
                    error[answer[j],answer[i]] += 1.0
            count[answer[i], answer[j]] += 1.0
            if not c:
                count[answer[j], answer[i]] += 1.0
    out /= (n*(n-1)/2.0)
    error /= count    
    return net.density, out, error
    

#getting the standard error of testing results
#when applying clustering algorithm to simulated network
def se_testing(size, prob, p, n):
    se = np.zeros(n)
    se_block = np.zeros((n, prob.shape[0], prob.shape[1]))
    for i in range(n):
        d, o, e = testing(size, prob, p)
        se[i] = o
        se_block[i] = e
    return se.mean(axis=0), se.std(axis=0)/np.sqrt(n), \
           se_block.mean(axis=0), se_block.std(axis=0)/np.sqrt(n)


#compute expected internal link density of a random 
#network from stochastic block model
def get_density(size, prob):
    k = len(size)
    numerator = 0.0
    total = np.sum(size)
    denominator = (total-1) * total
    for i in xrange(k):
        for j in xrange(k):
            if i == j:
                numerator += (size[i]-1) * size[i] * prob[i,i]
            if i != j:
                numerator += size[i] * size[j] * prob[i,j]
    return numerator/denominator
                
    
#choose parameter p automatically by sub-network
#size and link density
#size is an integer of sub_network size
#p0 is the internal link density
#k is a control parameter ranging from 2 to 4
def choose_p(size, p0, risk=0.025):
    if risk == 'ignore':
        return p0
    std = np.sqrt(p0*(1.0-p0)/((size-1)*(1-risk)))
    cut_off = norm.ppf(1-risk)
    phi = norm.pdf(cut_off)
    mu_adj = p0 - phi/(1-risk)*std
    """
    std_adj = std*np.sqrt(1-cut_off*norm.pdf(cut_off)/(1-risk)-
                          (norm.pdf(cut_off)/(1-risk))**2)
    """
    out = mu_adj - cut_off*std
    return max(out, 0)
    



#compute the lower bound for p
#to avoid merging two clusters with
#inter-cluster link probability p12
#size1 and size2 are the sizes of the
#two clusters
#k is a control parameter ranging from 2 to 4
def lower_p(size1, size2, p12, k=1.96):
    out = p12 + k*np.sqrt(p12*(1.0-p12)/(size1*size2))
    return min(1.0, out)
    

#my implementation of the normalized mutual information 
#between two clustering
#index is the correct clustering list
#cluster is the clustering list from algorithm
#version 0 is from 
def my_nmi(index, cluster, version=0):
    confuse = np.zeros((len(index), len(cluster)))
    numerator = 0
    ind = [set(item) for item in index]
    clu = [set(item) for item in cluster]
    for i, ind_item in enumerate(ind):
        for j, clu_item in enumerate(clu):
            confuse[i,j] = float(len(ind_item.intersection(clu_item)))
    total = confuse.sum()
    total_ind = confuse.sum(axis=1)
    total_clu = confuse.sum(axis=0)
    for (i,j), count in np.ndenumerate(confuse):
        if confuse[i,j] == 0:
            continue
        numerator += confuse[i,j] * np.log(total*confuse[i,j]/
                     (total_ind[i]*total_clu[j]))
    d1 = np.sum(total_ind*np.log(1.0*total_ind/total))
    d2 = np.sum(total_clu*np.log(1.0*total_clu/total))
    if version == 0:
        return -2.0*numerator/(d1+d2)
    if version == 1:
        return numerator/np.sqrt(d1*d2)


#alternative way to implement normalized mutual informtion
#labels_true and labels_pred are 1D numpy array of membership
def my_nmi_alt(labels_true, labels_pred, version=0):
    nows_n = len(set(labels_true))
    cols_n = len(set(labels_pred))
    confuse = np.zeros((nows_n, cols_n))
    numerator = 0
    rows_dict = {}
    cols_dict = {}
    rows_ind = 0
    cols_ind = 0
    for i, j in zip(labels_true, labels_pred):
        if i not in rows_dict:
            rows_dict[i] = rows_ind
            rows_ind += 1
        if j not in cols_dict:
            cols_dict[j] = cols_ind
            cols_ind += 1
        confuse[rows_dict[i],cols_dict[j]] += 1
    total = confuse.sum()
    total_ind = confuse.sum(axis=1)
    total_clu = confuse.sum(axis=0)
    for (i,j), count in np.ndenumerate(confuse):
        if confuse[i,j] == 0:
            continue
        numerator += confuse[i,j] * np.log(total*confuse[i,j]/
                     (total_ind[i]*total_clu[j]))
    d1 = np.sum(total_ind*np.log(1.0*total_ind/total))
    d2 = np.sum(total_clu*np.log(1.0*total_clu/total))
    if version == 0:
        return -2.0*numerator/(d1+d2)
    if version == 1:
        return numerator/np.sqrt(d1*d2)
    

def read_mcmc(infile, delimiter=','):
    result = []
    row = 0
    with open(infile, 'rb') as f:
        for line in f:
            if row == 0:
                row += 1
                continue
            line = line.strip()
            entries = line.split(delimiter)
            weight = entries[2]
            if row % 2 == 1:
                check = [weight]
            elif row % 2 == 0:
                check.append(weight)
                result.append(check.index(max(check)))
            row += 1
    return np.array(result)
    

#get the cluster membership by Bayesian method
dolphin = read_mcmc('dolphin_cluster.csv')

#get the cluster membership by betweenness method
dolphin_bet = np.zeros(62, dtype=int)
check_set = set([0,2,10,28,30,42,47])
for i in xrange(62):
    if i in check_set:
        dolphin_bet[i] = (dolphin[i]+1) % 2
    else:
        dolphin_bet[i] = dolphin[i]


#making plot of Bayesian network clustering
net.plot_network(title='Bayesian Dolphin Network Clustering',
                 membership = dolphin, outfile='graph/dolphin_bayes.png',
                 width=900, height=600)


#making plot of Betweenness network clustering
net.plot_network(title='Betweenness Dolphin Network Clustering',
                 membership = dolphin_bet, outfile='graph/dolphin_bet.png',
                 width=900, height=600)


    
size0 = np.array([40])
prob0 = np.array([[0.1]])
se_testing(size0, prob0, 'auto', 100)
aa, ii = simulate(size0, prob0)
net = Clustering(aa)
net.mod_dfs()
net.plot_network(title='Modularity-based Clustering on network from G(40, 0.1)')


ss = size0[1:]
pp = prob0[1:,1:]
se_testing(ss, pp, 0.24, 100)




size1 = np.array([100, 10, 10])
prob1 = np.array([[0.2, 0.05, 0.05],
                  [0.05, 0.5, 0.05],
                  [0.05, 0.05, 0.5]])
nmi_testing(size1, prob1, 'modularity', 100)
nmi_testing(size1, prob1, 0.11, 100)
se_testing(size1, prob1, 'modularity', 100) 

se_testing(size1, prob1, 0.09, 100)
                  
size2 = np.array([800, 400, 50, 20])
prob2 = np.array([[0.1,  0.04, 0.01, 0.01],
                  [0.04, 0.15, 0.01, 0.01],
                  [0.01, 0.01, 0.4,  0.02],
                  [0.01, 0.01, 0.02, 0.6]])
se_testing(size2, prob2, 0.06, 100)   
nmi_testing(size2, prob2, 0.06, 100)
nmi_testing(size2, prob2, 'auto', 100)
                  

size3 = np.array([3000, 2000, 1000,
                  400,  200,  100,
                  100,  100,  80, 20])
prob3 = np.zeros((10,10)) + 0.005
prob3[0,0] = 0.08
prob3[1,1] = 0.09
prob3[2,2] = 0.1
prob3[3,3] = 0.15
prob3[4,4] = 0.2
prob3[5,5] = 0.25
prob3[6,6] = 0.25
prob3[7,7] = 0.25
prob3[8,8] = 0.3
prob3[9,9] = 0.7
nmi_testing(size3, prob3, 'auto', 5)
nmi_testing(size3, prob3, 0.06, 1)


size4 = np.array([100, 20, 20])
prob4 = np.array([[0.15, 0.01, 0.01],
                  [0.01, 0.8, 0.02],
                  [0.01, 0.02, 0.8]])




size5 = np.array([100, 20, 20])
prob5 = np.array([[0.2, 0.05, 0.05],
                  [0.05, 0.6, 0.12],
                  [0.05, 0.12, 0.6]])
se_testing(size5, prob5, 'modularity', 100) 


size6 = np.array([40,40,40])
prob6 = np.array([[0.2, 0.01, 0.01],
                  [0.01, 0.2, 0.01],
                  [0.01, 0.01, 0.2]])
nmi_testing(size, prob, 'auto', 100)




#size6 = np.array([32, 32, 32, 32])
#prob6 = np.array([[0.33, 0.05, 0.05, 0.05],
#                  [0.05, 0.33, 0.05, 0.05],
#                  [0.05, 0.05, 0.33, 0.05],
#                  [0.05, 0.05, 0.05, 0.33]])
#nmi_testing(size6, prob6, 'modularity', 100)


c = np.zeros((120, 120))
simulate_cluster(c, range(100), 0.2)
simulate_cluster(c, range(100,110), 0.5)
simulate_cluster(c, range(110,120), 0.5)
simulate_link(c, range(100), range(100,110), 0.05)
simulate_link(c, range(100), range(110, 120), 0.05)
simulate_link(c, range(100, 110), range(110,120), 0.05)
net = Clustering(c)
net.partition(range(120), 0.12, 1)
        
                
a = np.zeros((1200,1200))
simulate_cluster(a, range(800), 0.1)
simulate_cluster(a, range(800, 1180), 0.15)
simulate_cluster(a, range(1180, 1200), 0.6)
simulate_link(a, range(800), range(800, 1180), 0.01)
simulate_link(a, range(800), range(1180, 1200), 0.01)
simulate_link(a, range(800, 1180), range(1180, 1200), 0.03)

net = Clustering(a)
net.partition(range(1200), 0.06, 1)
 
b = np.zeros((7000, 7000))
simulate_cluster(b, range(6000), 0.1)
simulate_cluster(b, range(6000, 6500), 0.12)
simulate_cluster(b, range(6500, 6980), 0.15)
simulate_cluster(b, range(6980, 7000), 0.6)

simulate_link(b, range(6000), range(6000, 6500), 0.01)
simulate_link(b, range(6000), range(6500, 6980), 0.01)
simulate_link(b, range(6000), range(6980, 7000), 0.01)
simulate_link(b, range(6000, 6500), range(6500, 6980), 0.01)
simulate_link(b, range(6000, 6500), range(6980, 7000), 0.01)
simulate_link(b, range(6500, 6980), range(6980, 7000), 0.01)

net = Clustering(b)
net.partition(range(7000), 0.06, 1)


d = np.loadtxt('dolphin.txt')
net = Clustering(d)
net.mod_partition(range(62))
net.auto_partition(range(62))
net.partition(range(62), 0.15, 6)

d1 = np.loadtxt('karate.csv', delimiter=',')
net = Clustering(d1)
net.mod_partition(range(34))

net.auto_partition(range(34), error=0.025)
net.auto_partition(range(34))

size, density, parameter = np.loadtxt('parameter.csv', delimiter=',', 
                                      usecols=(0,2,3), skiprows=1,
                                      unpack=True)
                                      
result = []
for n, d in zip(size, density):
    n = int(n)
    pp = choose_p(n, d, risk=0.025)
    ee = test_error1(n, d, pp, 100)[0]
    result.append(ee)
result = np.array(result).reshape((len(result),1))

parm = []
for n, d in zip(size, density):
    parm.append(choose_p(n, d, risk=0.025))
parm = np.array(parm).reshape((len(parm),1))

size=size.reshape((len(size),1))
density=density.reshape((len(density),1))

total = np.concatenate((size, density,parm,result), axis=1)


for n, d, p in zip(size, density, result):
    print n, d, choose_p(n, d, risk=0.025), p
    
np.savetxt('parm.csv', total, 
           header='size,density,parm,result',delimiter=',')
    



d1 = np.loadtxt('karate.csv', delimiter=',')
d_seq = d1.sum(axis=0).astype(int)

d_seq = np.array([5,6,7,8,9,10,11,12])
for i in range(10):
    G = nx.expected_degree_graph(d_seq)
    aa = nx.adjacency_matrix(G)
    aa = aa.todense()
    aa = np.array(aa)
    net = Clustering(aa)
    net.mod_partition(range(d_seq.shape[0]))
    print net.density
    

aa, ii = simulate(size4, prob4)
aa = aa.tocoo()
aa_edgelist = [(i,j) for i,j in zip(aa.row, aa.col) if i > j]
net = Clustering(aa)
net.auto_dfs()
net.get_membership()
cluster_lab = net.membership.astype(int)
color_dict={0:'red', 1:'blue', 2:'green', 3:'yellow', 4:'orange', 5:'grey',
            6:'cyan', 7:'white', 8:'black', 9:'purple', 10:'pink'}
gg = ig.Graph(aa_edgelist)
gg.vs['color'] = [color_dict[id] for id in cluster_lab]
gg.vs['size'] = 8
lay_fr = gg.layout('fr')
pp_fr = ig.plot(gg, layout=lay_fr)
pp_fr.show()

lay_kk = gg.layout('kk')
pp_kk = ig.plot(gg, layout=lay_kk)

lay_lgl = gg.layout('lgl')
pp_lgl = ig.plot(gg, layout=lay_lgl)
pp_lgl.show()

g.vs['size'] = 20
g.vs['color'] = [color_list[id] for id in cd.membership]
lay_kk = g.layout('kk')

pp_kk = ig.plot(g, target='haha.pdf',margin=(20,50,20,20), layout=lay_kk)
pp_kk.redraw()
ctx = cairo.Context(pp_kk.surface)
ctx.set_font_size(20)
drawer = TextDrawer(ctx, "Test title", halign=TextDrawer.CENTER)
drawer.draw_at(3, 30, width=600)
pp_kk.show()

pp_kk.show()
pp_lgl.show()

#generate graph from stochastic block model
gg = ig.Graph.SBM(n=120, pref_matrix=list(prob1), 
                  block_sizes=list(size1))

#gg = ig.Graph.Erdos_Renyi(100, 0.1)

cc = gg.community_leading_eigenvector()

adjlist = gg.get_adjlist()
aa = adjlist_to_adj(adjlist)
net = Clustering(aa)

gg.vs['color'] = [color_dict[id] for id in cc.membership]
gg.vs['size'] = 8
lay_fr = gg.layout('fr')
pp_fr = ig.plot(gg, layout=lay_fr)
lay_kk = gg.layout('kk')
pp_kk = ig.plot(gg, layout=lay_kk)
lay_lgl = gg.layout('lgl')
pp_lgl = ig.plot(gg, layout=lay_lgl)

pp_fr.show()
pp_kk.show()
pp_lgl.show()
 
 
 
lay_kk = g.layout('kk')
pp_kk = ig.plot(g, layout=lay_kk)
pp_kk.show()
    
    
