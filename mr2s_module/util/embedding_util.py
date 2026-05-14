import networkx as nx
import minorminer
import dwave_networkx as dnx
from dimod import BinaryQuadraticModel

from mr2s_module.domain import EmbeddingEstimate

_pegasus_graph: nx.Graph | None = None


def _get_pegasus_p16() -> nx.Graph:
    global _pegasus_graph
    if _pegasus_graph is None:
        _pegasus_graph = dnx.pegasus_graph(16)
    return _pegasus_graph


def estimate_required_qubits(bqm: BinaryQuadraticModel, target_graph: nx.Graph = _get_pegasus_p16()) -> EmbeddingEstimate:
    """minorminer + Pegasus P16 토폴로지를 사용하여 필요 물리 큐빗 수를 추정한다."""
    source_graph = nx.Graph()
    source_graph.add_nodes_from(bqm.variables)
    source_graph.add_edges_from(bqm.quadratic)

    if source_graph.number_of_nodes() == 0:
        return EmbeddingEstimate(
            num_logical_variables=0,
            num_quadratic_couplings=0,
            num_physical_qubits=0,
            max_chain_length=0,
            embedding={},
        )

    if source_graph.number_of_edges() == 0:
        target_nodes = list(target_graph.nodes)
        if source_graph.number_of_nodes() > len(target_nodes):
            raise RuntimeError("Pegasus P16 토폴로지에 임베딩을 찾을 수 없습니다.")
        embedding = {
            variable: [target_nodes[index]]
            for index, variable in enumerate(source_graph.nodes)
        }
        return EmbeddingEstimate(
            num_logical_variables=len(bqm.variables),
            num_quadratic_couplings=0,
            num_physical_qubits=len(embedding),
            max_chain_length=1,
            embedding=embedding,
        )

    embedding = minorminer.find_embedding(
        source_graph.edges(),
        target_graph.edges(),
    )

    if not embedding:
        raise RuntimeError("Pegasus P16 토폴로지에 임베딩을 찾을 수 없습니다.")

    num_physical_qubits = sum(len(chain) for chain in embedding.values())
    max_chain_length = max(len(chain) for chain in embedding.values())

    return EmbeddingEstimate(
        num_logical_variables=len(bqm.variables),
        num_quadratic_couplings=len(bqm.quadratic),
        num_physical_qubits=num_physical_qubits,
        max_chain_length=max_chain_length,
        embedding={variable: list(chain) for variable, chain in embedding.items()},
    )
