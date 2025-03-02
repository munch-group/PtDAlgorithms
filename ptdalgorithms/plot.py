
import graphviz
import random
from collections import defaultdict

def random_color():
    return '#'+''.join(random.sample('0123456789ABCDEF', 6))


def format_rate(rate):
    if rate == round(rate):
        return f"{rate:.2f}"
    else:
        return f"{rate:.2e}"


def plot_graph(graph, constraint=True, subgraphs=False, ranksep=1, 
               nodesep=0.5, splines='splines', 
               subgraphfun=lambda state, index: ','.join(map(str, state[:-1])), size=(6, 6), 
               fontsize=10, rankdir="LR", align=False, nodecolor='white', 
               edgecolor='rainbow', penwidth=1):

    graph_attr = dict(
        compound='true',
        newrank='true',
        pad='0.5',
        ranksep=str(ranksep),
        nodesep=str(nodesep),
        bgcolor='transparent',
        rankdir=rankdir,
        splines=splines,
        size=f'{size[0]},{size[1]}',
        fontname="Helvetica,Arial,sans-serif",
        ratio="fill",
    )
    node_attr = dict(
        style='filled',
        color='black',
    	fontname="Helvetica,Arial,sans-serif", 
        fontsize=str(fontsize), 
        fillcolor=str(nodecolor),
    )
    edge_attr = dict(
        constraint='true' if constraint else 'false',
        style='filled',
        labelfloat='false', 
        labeldistance='0',
    	fontname="Helvetica,Arial,sans-serif", 
        fontsize=str(fontsize), 
        penwidth=str(penwidth),
    )
    dot = graphviz.Digraph(
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
    )
    for i in range(graph.vertices_length()):
        vertex = graph.vertex_at(i)
        for edge in vertex.edges():
            if edgecolor == 'rainbow':
                color = random_color()
            else:
                 color = edgecolor
            dot.edge(str(vertex.index()), str(edge.to().index()), 
                   xlabel=format_rate(edge.weight()), color=color, fontcolor=color)

    subgraph_attr = dict(
        rank='same',
        style='filled',
        color='whitesmoke',
    )
    subg = defaultdict(list)
    for i in range(graph.vertices_length()):
        vertex = graph.vertex_at(i)
        if i == 0:
            dot.node(str(vertex.index()), 'S', 
                     style='filled', color='black', fillcolor='whitesmoke')
        elif not vertex.edges():
            dot.node(str(vertex.index()), ','.join(map(str, vertex.state())), 
                     style='filled', color='black',fillcolor='whitesmoke')
        elif subgraphs:
            subg[f'cluster_{subgraphfun(vertex.state())}'].append(i)
        else:
            dot.node(str(vertex.index()), ','.join(map(str, vertex.state())))
    if subgraphs:
        for sglabel in subg:
            with dot.subgraph(name=sglabel, graph_attr=subgraph_attr) as c:
                for i in subg[sglabel]:
                    vertex = graph.vertex_at(i)
                    c.node(str(vertex.index()), ','.join(map(str, vertex.state())))
    return dot
