import networkx as nx


class BaseDiGraph(nx.DiGraph):
    """
    Base wrapper for networkx directed graph
    """

    def __init__(self, **attr: object):
        super().__init__(**attr)

    def save_svg(self, path: str):
        from networkx.drawing.nx_agraph import to_agraph
        graph = to_agraph(self)
        graph.draw(path, format='svg', prog='dot')