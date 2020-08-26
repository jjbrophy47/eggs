"""
Module that puts all connected components into separate groups.
"""
import pandas as pd
import networkx as nx


# public
def create_clusters(target_priors, relations_dict, target_col='com_id',
                    max_size=40000, max_edges=-1, logger=None):
    """
    Given the target IDs and the relational information, create clusters
    of connected components to do inference over.
    """

    prior_df = pd.DataFrame(target_priors, columns=[target_col, 'ind_yhat'])

    graph, components = _build_networkx_graph(prior_df, relations_dict, target_col=target_col, logger=logger)

    print(len(components))

    if max_edges > 0:
        components = _partition_large_components(components, graph, max_edges,
                                                 target_col=target_col, logger=logger)
        print(len(components))

    components = _process_components(components, graph, target_col=target_col, logger=logger)
    components = _filter_redundant_components(prior_df, components, target_col=target_col, logger=logger)
    components = _remove_single_edge_hubs(components, graph, target_col=target_col, logger=logger)
    components = _remove_single_node_components(components, logger=logger)
    clusters = _consolidate(components, max_size=max_size, logger=logger)

    return clusters


# private
def _build_networkx_graph(prior_df, relations_dict, target_col='com_id', logger=None):
    """
    Builds relational graph to find connected components.
    """
    if logger:
        logger.info('building networkx graph...')

    graph = nx.Graph()

    # create edges from target nodes to hub nodes
    for relation_col, connection_list in relations_dict.items():

        for hub_id, target_id in connection_list:
            if target_id in prior_df[target_col].unique():
                graph.add_edge('{}-{}'.format(target_col, target_id), '{}-{}'.format(relation_col, hub_id))

    components = list(nx.connected_components(graph))

    return graph, components


def _partition_large_components(components, graph, max_edges,
                                target_col='com_id', logger=None):
    """
    Break up extremely large connected components to make inference more tractable.
    """
    if logger:
        logger.info('partitioning very large subgraphs...')

    new_components = []
    for component in components:
        hub_nodes = {node for node in component if target_col not in str(node)}
        num_edges = sum([graph.degree(x) for x in hub_nodes])

        if num_edges >= max_edges:
            new_component = set()
            new_edges = 0

            for hub_node in hub_nodes:
                hub_edges = graph.degree(hub_node)
                neighbors = set(graph.neighbors(hub_node))

                # keep adding
                if hub_edges + new_edges <= max_edges:
                    new_component.add(hub_node)
                    new_component.update(neighbors)
                    new_edges += hub_edges

                # new is full
                elif hub_edges + new_edges > max_edges:

                    # nothing in new
                    if new_edges == 0:
                        new_component = {hub_node}
                        new_component.update(neighbors)
                        new_edges = hub_edges

                    # flush, then start new component
                    else:
                        new_components.append(new_component)
                        new_component = {hub_node}
                        new_component.update(neighbors)
                        new_edges = hub_edges

            # take care of leftovers
            if len(new_component) > 0:
                new_components.append(new_component)
        else:
            new_components.append(component)

    return new_components


def _process_components(components, graph, target_col='com_id', logger=None):
    """
    Extract the IDs of each message and hub node. Also extract
    the relations and number of edges in each component.
    """
    if logger:
        logger.info('processing each component...')

    subgraphs = []
    for component in components:
        relations = set()
        n_edges = 0

        msg_nodes = {node for node in component if target_col in str(node)}
        hub_nodes = {node for node in component if target_col not in str(node)}

        for hub_node in hub_nodes:
            relations.add(hub_node.split('-')[0])
            n_edges += graph.degree(hub_node)

        subgraphs.append((msg_nodes, hub_nodes, relations, n_edges))

    return subgraphs


def _filter_redundant_components(prior_df, components, target_col='com_id', logger=None):
    """
    Filters out subgraphs whose message predictions are already
    either ALL ones or zeros.
    """
    if logger:
        logger.info('filtering redundant components...')

    new_components = []
    for msg_nodes, hub_nodes, relations, n_edges in components:
        redundant = False

        # get average prediction score from all messages in the cluster
        component_target_ids = [node.split('-')[1] for node in msg_nodes]
        temp_df = prior_df[prior_df[target_col].isin(component_target_ids)]
        yhat_mean = temp_df['ind_yhat'].mean()
        pred_sum = sum(list(temp_df['ind_yhat']))

        # remove cluster from inference if predictions are already extreme
        if (pred_sum == 0 and yhat_mean == 0) or (pred_sum == len(temp_df) and yhat_mean == 1):
            redundant = True

        if not redundant:
            new_components.append((msg_nodes, hub_nodes, relations, n_edges))

    return new_components


def _remove_single_edge_hubs(components, graph, target_col='com_id', logger=None):
    """
    Remove components that only have a single edge.
    """
    if logger:
        logger.info('removing hub nodes with a single edge in each subgraph...')

    new_components = []
    for msg_nodes, hub_nodes, relations, n_edges in components:

        all_nodes = msg_nodes.union(hub_nodes)
        new_hub_nodes = hub_nodes.copy()
        component = graph.subgraph(all_nodes)

        for node in component:
            if target_col not in str(node):
                hub_deg = component.degree(node)

                if hub_deg == 1:
                    new_hub_nodes.remove(node)
                    n_edges -= 1

        new_components.append((msg_nodes, new_hub_nodes, relations, n_edges))

    return new_components


def _remove_single_node_components(components, logger=None):
    """
    Remove nodes with no connections.
    """
    if logger:
        logger.info('compiling single node subgraphs...')

    new_components = []
    for msg_nodes, hub_nodes, relations, n_edges in components:
        if n_edges > 1:
            new_components.append((msg_nodes, hub_nodes, relations, n_edges))

    return new_components


def _consolidate(components, max_size=40000, div=2, logger=None):
    """
    Combine components into clusters to reduce the total number of
    times inference must be invoked.
    """
    if logger:
        logger.info('consolidating components...')

    # result containers
    clusters = []
    new_msg_nodes = set()
    new_hub_nodes = set()
    new_relations = set()
    new_n_edges = 0

    for msg_nodes, hub_nodes, relations, n_edges in components:
        size = int(len(new_msg_nodes) / div) + int(len(msg_nodes) / div)
        size += new_n_edges + n_edges

        # keep adding to new
        if size < max_size:
            new_msg_nodes.update(msg_nodes)
            new_relations.update(relations)
            new_hub_nodes.update(hub_nodes)
            new_n_edges += n_edges

        # cluster is full, start a new cluster
        else:

            # nothing in current cluster
            if len(new_msg_nodes) == 0:
                clusters.append((msg_nodes, hub_nodes, relations, n_edges))

            # cluster has leftovers, flush
            else:
                clusters.append((new_msg_nodes, new_hub_nodes, new_relations, new_n_edges))
                new_msg_nodes = msg_nodes
                new_hub_nodes = hub_nodes
                new_relations = relations
                new_n_edges = n_edges

    # add any leftovers to a new cluster
    if len(new_msg_nodes) > 0:
        clusters.append((new_msg_nodes, new_hub_nodes, new_relations, new_n_edges))

    return clusters


def _print_cluster_statistics(clusters, logger=None):
    """
    Print the number of clusters, and the total number of message nodes,
    hub nodes, and number of edges.
    """
    tot_m, tot_h, tot_e = 0, 0, 0

    for ids, hubs, rels, edges in clusters:
        tot_m += len(ids)
        tot_h += len(hubs)
        tot_e += edges

    t = (len(clusters), tot_m, tot_h, tot_e)

    if logger:
        logger.info('clusters: %d, msgs: %d, hubs: %d, edges: %d' % t)
