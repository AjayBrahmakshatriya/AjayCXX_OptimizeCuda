#include "graph.h"


int main(int argc, char* argv[]) {
	assert(argc >= 2 && "Usage: executable <graph name>");
	graph_t g = load_graph(argv[1]);
	return 0;
}
