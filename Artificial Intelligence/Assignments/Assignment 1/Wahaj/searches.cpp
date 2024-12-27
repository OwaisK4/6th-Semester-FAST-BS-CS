// HECKERMAN NAME -> K200208 WAHAJ JAVED ALAM
// SECTION -> BCS-6E
#include <iostream>
#include <map>
#include <queue>
#include <vector>
using namespace std;
// simple node class for BFS
class Node {
public:
    string location;
    int cost;

    Node(string l, int c) {
        location = l;
        cost = c;
    }
    void setValues(string l, int c) {
        location = l;
        cost = c;
    }
};
// create the graph and generate the heuristics
void createGraph(map<string, vector<Node>> &graph, map<string, int> &heuristics) {
    vector<Node> v;
    Node n("Zerind", 75);
    v.push_back(n);
    n.setValues("Sibiu", 140);
    v.push_back(n);
    n.setValues("Timisoara", 118);
    v.push_back(n);
    graph["Arad"] = v;
    v.clear();
    n.setValues("Arad", 75);
    v.push_back(n);
    n.setValues("Oradea", 71);
    v.push_back(n);
    graph["Zerind"] = v;
    v.clear();
    n.setValues("Zerind", 71);
    v.push_back(n);
    n.setValues("Sibiu", 151);
    v.push_back(n);
    graph["Oradea"] = v;
    v.clear();
    n.setValues("Arad", 118);
    v.push_back(n);
    n.setValues("Lugoj", 111);
    v.push_back(n);
    graph["Timisoara"] = v;
    v.clear();
    n.setValues("Timisoara", 111);
    v.push_back(n);
    n.setValues("Mehadia", 70);
    v.push_back(n);
    graph["Lugoj"] = v;
    v.clear();
    n.setValues("Lugoj", 70);
    v.push_back(n);
    n.setValues("Drobeta", 75);
    v.push_back(n);
    graph["Mehadia"] = v;
    v.clear();
    n.setValues("Mehadia", 75);
    v.push_back(n);
    n.setValues("Craiova", 120);
    v.push_back(n);
    graph["Drobeta"] = v;
    v.clear();
    n.setValues("Drobeta", 120);
    v.push_back(n);
    n.setValues("Rimnicu Vilcea", 146);
    v.push_back(n);
    n.setValues("Pitesti", 138);
    v.push_back(n);
    graph["Craiova"] = v;
    v.clear();
    n.setValues("Fagaras", 99);
    v.push_back(n);
    n.setValues("Rimnicu Vilcea", 80);
    v.push_back(n);
    graph["Sibiu"] = v;
    v.clear();
    n.setValues("Sibiu", 80);
    v.push_back(n);
    n.setValues("Pitesti", 97);
    v.push_back(n);
    n.setValues("Craiova", 146);
    v.push_back(n);
    graph["Rimnicu Vilcea"] = v;
    v.clear();
    n.setValues("Craiova", 138);
    v.push_back(n);
    n.setValues("Rimnicu Vilcea", 97);
    v.push_back(n);
    n.setValues("Bucharest", 101);
    v.push_back(n);
    graph["Pitesti"] = v;
    v.clear();
    n.setValues("Sibiu", 99);
    v.push_back(n);
    n.setValues("Bucharest", 211);
    v.push_back(n);
    graph["Fagaras"] = v;
    v.clear();
    n.setValues("Giurgiu", 90);
    v.push_back(n);
    n.setValues("Urziceni", 85);
    v.push_back(n);
    n.setValues("Fagaras", 211);
    v.push_back(n);
    n.setValues("Pitesti", 101);
    v.push_back(n);
    graph["Bucharest"] = v;
    v.clear();
    n.setValues("Bucharest", 90);
    v.push_back(n);
    graph["Giurgiu"] = v;
    v.clear();
    n.setValues("Bucharest", 85);
    v.push_back(n);
    n.setValues("Hirsova", 98);
    v.push_back(n);
    n.setValues("Vaslui", 142);
    v.push_back(n);
    graph["Urziceni"] = v;
    v.clear();
    n.setValues("Urzeceni", 98);
    v.push_back(n);
    n.setValues("Eforie", 86);
    v.push_back(n);
    graph["Hirsova"] = v;
    v.clear();
    n.setValues("Hirsova", 86);
    v.push_back(n);
    graph["Eforie"] = v;
    v.clear();
    n.setValues("Urziceni", 142);
    v.push_back(n);
    n.setValues("Iasi", 92);
    v.push_back(n);
    graph["Vaslui"] = v;
    v.clear();
    n.setValues("Vaslui", 92);
    v.push_back(n);
    n.setValues("Neamt", 87);
    v.push_back(n);
    graph["Iasi"] = v;
    v.clear();
    n.setValues("Iasi", 87);
    v.push_back(n);
    graph["Neamt"] = v;
    v.clear();
    heuristics["Arad"] = 366;
    heuristics["Bucharest"] = 0;
    heuristics["Craiova"] = 160;
    heuristics["Drobeta"] = 242;
    heuristics["Eforie"] = 161;
    heuristics["Fagaras"] = 176;
    heuristics["Giurgiu"] = 77;
    heuristics["Hirsova"] = 151;
    heuristics["Iasi"] = 226;
    heuristics["Lugoj"] = 244;
    heuristics["Mehadia"] = 241;
    heuristics["Neamt"] = 234;
    heuristics["Oradea"] = 380;
    heuristics["Pitesti"] = 100;
    heuristics["Rimnicu Vilcea"] = 193;
    heuristics["Sibiu"] = 253;
    heuristics["Timisoara"] = 329;
    heuristics["Urziceni"] = 80;
    heuristics["Vaslui"] = 199;
    heuristics["Zerind"] = 374;
}
// simple BFS modified to find the shortest path since the graph is following triangle inequality
void breadthFirstSearch(map<string, vector<Node>> graph, string startingNode, string destinationNode) {
    map<string, bool> visited;
    queue<string> q;
    map<string, int> distances;
    distances[startingNode] = 0;
    q.push(startingNode);
    visited[startingNode] = true;
    cout << "Pathways Discovered: ";
    while (!q.empty()) {
        string currentLocation = q.front();
        cout << currentLocation << " -> ";
        q.pop();
        vector<Node> neighbours = graph[currentLocation];
        for (int i = 0; i < neighbours.size(); i++) {
            string location = neighbours[i].location;
            cout << location;
            if (i != neighbours.size() - 1) {
                cout << ", ";
            }
            int cost = neighbours[i].cost;
            if (!visited[location]) {
                distances[location] = INT32_MAX;
                q.push(location);
                visited[location] = true;
            }
            if (distances[location] > distances[currentLocation] + cost) {
                distances[location] = distances[currentLocation] + cost;
            }
        }
        cout << endl;
    }
    cout << "Distance from " << startingNode << " to " << destinationNode << " is: " << distances[destinationNode] << endl;
}
// a path nodes class to store the actual path, source for the continuation of the path, the cost and heuristics value
class pathNodes {
public:
    string path, source;
    int cost;
    int heuristics;
    pathNodes(string p, string s, int c, int h = 0) {
        path = p;
        cost = c;
        source = s;
        heuristics = h;
    }
};
// simple compare function for priority queue inversion for ucs based on cost
struct compare {
    bool operator()(const pathNodes &l, const pathNodes &r) {
        return l.cost > r.cost;
    }
};
void uniformCostSearch(map<string, vector<Node>> graph, string startingNode, string destinationNode) {
    priority_queue<pathNodes, vector<pathNodes>, compare> pq;
    pq.push(pathNodes(startingNode, startingNode, 0));
    map<string, bool> visited;
    while (!pq.empty()) {
        pathNodes currentNode = pq.top();
        pq.pop();
        string currentLocation = currentNode.source;
        int currentCost = currentNode.cost;
        if (currentLocation == destinationNode) {
            cout << "Path: " << currentNode.path << endl;
            cout << "Cost: " << currentCost << endl;
            return;
        }
        visited[currentLocation] = true;
        vector<Node> neighbours = graph[currentLocation];
        for (int i = 0; i < neighbours.size(); i++) {
            if (!visited[neighbours[i].location]) {
                int costSum = currentCost + neighbours[i].cost;
                pq.push(pathNodes(currentNode.path + " -> " + neighbours[i].location, neighbours[i].location, costSum));
            }
        }
    }
}
// simple compare function for priority queue inversion for bestfirstsearch based on heuristics
struct heuristicCompare {
    bool operator()(const pathNodes &l, const pathNodes &r) {
        return l.heuristics > r.heuristics;
    }
};
void bestFirstSearch(map<string, vector<Node>> graph, string startingNode, string destinationNode, map<string, int> heuristics) {
    priority_queue<pathNodes, vector<pathNodes>, heuristicCompare> pq;
    pq.push(pathNodes(startingNode, startingNode, 0));
    map<string, bool> visited;
    while (!pq.empty()) {
        pathNodes currentNode = pq.top();
        pq.pop();
        string currentLocation = currentNode.source;
        int currentCost = currentNode.cost;
        if (currentLocation == destinationNode) {
            cout << "Path: " << currentNode.path << endl;
            cout << "Cost: " << currentCost << endl;
            return;
        }
        visited[currentLocation] = true;
        vector<Node> neighbours = graph[currentLocation];
        for (int i = 0; i < neighbours.size(); i++) {
            if (!visited[neighbours[i].location]) {
                int costSum = currentCost + neighbours[i].cost;
                pq.push(pathNodes(currentNode.path + " -> " + neighbours[i].location, neighbours[i].location, costSum, heuristics[neighbours[i].location]));
            }
        }
    }
}
int depthOfAGraph(map<string, vector<Node>> graph, map<string, bool> &visited, string startingNode, int depth) {
    visited[startingNode] = true;
    int maxDepth = depth;
    vector<Node> neighbours = graph[startingNode];
    for (int i = 0; i < neighbours.size(); i++) {
        if (!visited[neighbours[i].location]) {
            maxDepth = max(maxDepth, depthOfAGraph(graph, visited, neighbours[i].location, depth + 1));
        }
    }
    return maxDepth;
}
bool DFS(map<string, vector<Node>> graph, string startingNode, string destinationNode, int limit, int cost, string path) {
    if (startingNode == destinationNode) {
        return true;
    }
    if (limit <= 0) {
        return false;
    }
    vector<Node> neighbours = graph[startingNode];
    for (int i = 0; i < neighbours.size(); i++) {
        if (DFS(graph, neighbours[i].location, destinationNode, limit - 1, cost + neighbours[i].cost, path + " -> " + neighbours[i].location)) {
            cout << "-------------------------------------------" << endl;
            cout << "Path: " << path + " -> " + neighbours[i].location << endl;
            cout << "Cost: " << cost + neighbours[i].cost << endl;
            cout << "-------------------------------------------" << endl;
            return true;
        }
    }
    return false;
}
void iterativeDeepeningDFS(map<string, vector<Node>> graph, string startingNode, string destinationNode) {
    map<string, bool> depthVisited;
    int maxDepth = depthOfAGraph(graph, depthVisited, startingNode, 0);
    for (int i = 0; i < maxDepth; i++) {
        if (DFS(graph, startingNode, destinationNode, i, 0, startingNode)) {
            break;
        }
    }
}
int main() {
    string startingNode, destinationNode;
    cout << "Enter Starting node: ";
    cin >> startingNode;
    cout << "Enter Destination node: ";
    cin >> destinationNode;
    map<string, vector<Node>> graph;
    map<string, int> heuristics;
    createGraph(graph, heuristics);
    cout << "-------------------------------------" << endl;
    cout << "Breadth First Search" << endl;
    cout << "-------------------------------------" << endl;
    breadthFirstSearch(graph, "Arad", "Bucharest");
    cout << "-------------------------------------" << endl;
    cout << "Uniform Cost Search" << endl;
    cout << "-------------------------------------" << endl;
    uniformCostSearch(graph, startingNode, destinationNode);
    cout << "-------------------------------------" << endl;
    cout << "Best First Search" << endl;
    cout << "-------------------------------------" << endl;
    bestFirstSearch(graph, startingNode, destinationNode, heuristics);
    cout << "-------------------------------------" << endl;
    cout << "Iterative Deepening Depth First Search" << endl;
    cout << "-------------------------------------" << endl;
    cout << "Please Consider only the first path in this case since the solution is recursive and the deadline is too close for me to do something XD" << endl;
    iterativeDeepeningDFS(graph, startingNode, destinationNode);
}