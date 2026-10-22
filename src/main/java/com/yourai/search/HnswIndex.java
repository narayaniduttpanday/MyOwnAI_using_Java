package com.yourai.search;

import java.util.*;

/**
 * Compact in-memory Hierarchical Navigable Small World graph index.
 * Uses layered greedy descent and ef-bounded best-first search. This is an
 * educational implementation; tune M/ef values and benchmark recall for real workloads.
 */
public final class HnswIndex {
    private static final int DEFAULT_M = 16;
    private static final int DEFAULT_EF_CONSTRUCTION = 200;
    private static final int DEFAULT_EF_SEARCH = 50;
    private static final int MAX_LEVEL = 16;

    private final int m;
    private final int efConstruction;
    private final int efSearch;
    private final Random random;
    private final DistanceMetric buildMetric;
    private final Map<Integer, Node> nodes = new LinkedHashMap<>();
    private Node entryPoint;
    private int maxLevel = -1;
    private int dimensions;

    private static final class Node {
        final VectorItem item;
        final int level;
        final List<Set<Integer>> neighbors;
        Node(VectorItem item, int level) {
            this.item = item; this.level = level;
            this.neighbors = new ArrayList<>(level + 1);
            for (int i = 0; i <= level; i++) neighbors.add(new LinkedHashSet<>());
        }
    }

    private record Candidate(Node node, double distance) {}

    public HnswIndex(Collection<VectorItem> items) { this(items, DistanceMetric.EUCLIDEAN, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, DEFAULT_EF_SEARCH, 42L); }

    public HnswIndex(Collection<VectorItem> items, DistanceMetric metric) { this(items, metric, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, DEFAULT_EF_SEARCH, 42L); }

    public HnswIndex(Collection<VectorItem> items, DistanceMetric metric, int m, int efConstruction, int efSearch, long seed) {
        if (m < 2) throw new IllegalArgumentException("M must be at least 2");
        if (efConstruction < m) throw new IllegalArgumentException("efConstruction must be >= M");
        if (efSearch < 1) throw new IllegalArgumentException("efSearch must be >= 1");
        this.m = m; this.efConstruction = efConstruction; this.efSearch = efSearch;
        this.buildMetric = Objects.requireNonNull(metric, "metric");
        this.random = new Random(seed);
        List<VectorItem> copy = new ArrayList<>(items);
        this.dimensions = copy.isEmpty() ? 0 : copy.get(0).embedding().size();
        for (VectorItem item : copy) {
            if (item.embedding().size() != dimensions) throw new IllegalArgumentException("All indexed vectors must have the same dimension");
            add(item);
        }
    }

    public synchronized void insert(VectorItem item) {
        Objects.requireNonNull(item, "item");
        if (nodes.containsKey(item.id())) throw new IllegalArgumentException("Duplicate vector ID: " + item.id());
        if (nodes.isEmpty()) dimensions = item.embedding().size();
        if (item.embedding().size() != dimensions) throw new IllegalArgumentException("All indexed vectors must have the same dimension");
        for (Double value : item.embedding()) {
            if (value == null || !Double.isFinite(value)) throw new IllegalArgumentException("Vector values must be finite numbers");
        }
        add(item);
    }

    public int size() { return nodes.size(); }
    public int topLayer() { return maxLevel; }
    public int efSearch() { return efSearch; }

    public List<BruteForceIndex.Neighbor> search(List<Double> query, int k, DistanceMetric metric) {
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        if (entryPoint == null) return List.of();
        if (metric != buildMetric) throw new IllegalArgumentException("This HNSW index was built for " + buildMetric.name().toLowerCase(Locale.ROOT));
        if (query == null || query.size() != dimensions) throw new IllegalArgumentException("Query dimension must match indexed vectors");
        Node current = entryPoint;
        double currentDistance = metric.distance(query, current.item.embedding());
        for (int layer = maxLevel; layer >= 1; layer--) {
            boolean changed;
            do {
                changed = false;
                for (int neighborId : current.neighbors.get(layer)) {
                    Node neighbor = nodes.get(neighborId);
                    double d = metric.distance(query, neighbor.item.embedding());
                    if (d < currentDistance || (Double.compare(d, currentDistance) == 0 && neighbor.item.id() < current.item.id())) {
                        current = neighbor; currentDistance = d; changed = true;
                    }
                }
            } while (changed);
        }
        List<Candidate> candidates = searchLayer(query, List.of(current), Math.max(k, efSearch), 0, metric);
        return candidates.stream().map(c -> new BruteForceIndex.Neighbor(c.node.item, c.distance))
            .sorted(Comparator.comparingDouble(BruteForceIndex.Neighbor::distance).thenComparingInt(n -> n.item().id()))
            .limit(k).toList();
    }

    public Map<String, Object> graphInfo() {
        List<Map<String, Object>> nodeInfo = new ArrayList<>();
        List<Map<String, Object>> edgeInfo = new ArrayList<>();
        int[] nodesPerLayer = new int[Math.max(0, maxLevel + 1)];
        int[] edgesPerLayer = new int[Math.max(0, maxLevel + 1)];
        for (Node node : nodes.values()) {
            nodeInfo.add(Map.of("id", node.item.id(), "metadata", node.item.metadata(), "category", node.item.category(), "maxLyr", node.level));
            for (int layer = 0; layer <= node.level; layer++) {
                nodesPerLayer[layer]++;
                for (int target : node.neighbors.get(layer)) {
                    edgesPerLayer[layer]++;
                    edgeInfo.add(Map.of("source", node.item.id(), "target", target, "layer", layer,
                        "src", node.item.id(), "dst", target, "lyr", layer));
                }
            }
        }
        Map<String, Object> info = new LinkedHashMap<>();
        info.put("topLayer", maxLevel);
        info.put("entryPoint", entryPoint == null ? -1 : entryPoint.item.id());
        info.put("nodeCount", nodes.size());
        info.put("nodesPerLayer", Arrays.stream(nodesPerLayer).boxed().toList());
        info.put("edgesPerLayer", Arrays.stream(edgesPerLayer).boxed().toList());
        info.put("nodes", nodeInfo);
        info.put("edges", edgeInfo);
        info.put("M", m);
        info.put("M0", 2 * m);
        info.put("efConstruction", efConstruction);
        info.put("efSearch", efSearch);
        info.put("buildMetric", buildMetric.name().toLowerCase(Locale.ROOT));
        info.put("note", "In-memory HNSW graph; insertions update the graph incrementally and deletions trigger an index rebuild in the store.");
        return info;
    }

    private void add(VectorItem item) {
        int level = randomLevel();
        Node inserted = new Node(item, level);
        if (entryPoint == null) {
            nodes.put(item.id(), inserted); entryPoint = inserted; maxLevel = level; return;
        }
        Node current = entryPoint;
        double currentDistance = distance(item, current.item);
        for (int layer = maxLevel; layer > level; layer--) {
            boolean changed;
            do {
                changed = false;
                for (int neighborId : current.neighbors.get(layer)) {
                    Node neighbor = nodes.get(neighborId);
                    double d = distance(item, neighbor.item);
                    if (d < currentDistance) { current = neighbor; currentDistance = d; changed = true; }
                }
            } while (changed);
        }
        nodes.put(item.id(), inserted);
        for (int layer = Math.min(level, maxLevel); layer >= 0; layer--) {
            List<Candidate> candidates = searchLayer(item.embedding(), List.of(current), efConstruction, layer, buildMetric);
            int maxConnections = maxConnections(layer);
            List<Node> selected = selectNeighbors(item, candidates, maxConnections);
            inserted.neighbors.get(layer).addAll(selected.stream().map(n -> n.item.id()).toList());
            for (Node neighbor : selected) {
                neighbor.neighbors.get(layer).add(item.id());
                pruneConnections(neighbor, layer);
            }
            if (!candidates.isEmpty()) current = candidates.get(0).node;
        }
        if (level > maxLevel) { entryPoint = inserted; maxLevel = level; }
    }

    private void pruneConnections(Node node, int layer) {
        Set<Integer> links = node.neighbors.get(layer);
        int maxConnections = maxConnections(layer);
        if (links.size() <= maxConnections) return;
        List<Node> ordered = links.stream().map(nodes::get).filter(Objects::nonNull)
            .sorted(Comparator.comparingDouble((Node n) -> distance(node.item, n.item)).thenComparingInt(n -> n.item.id())).toList();
        links.clear();
        for (Node neighbor : ordered.subList(0, Math.min(maxConnections, ordered.size()))) links.add(neighbor.item.id());
    }

    private List<Node> selectNeighbors(VectorItem query, List<Candidate> candidates, int limit) {
        List<Candidate> ordered = new ArrayList<>(candidates);
        ordered.sort(Comparator.comparingDouble(Candidate::distance).thenComparingInt(c -> c.node.item.id()));
        List<Node> selected = new ArrayList<>();
        List<Node> deferred = new ArrayList<>();
        for (Candidate candidate : ordered) {
            boolean diverse = true;
            for (Node chosen : selected) {
                if (distance(candidate.node.item, chosen.item) < candidate.distance) { diverse = false; break; }
            }
            if (diverse && selected.size() < limit) selected.add(candidate.node);
            else deferred.add(candidate.node);
        }
        for (Node node : deferred) {
            if (selected.size() >= limit) break;
            if (!selected.contains(node)) selected.add(node);
        }
        return selected;
    }

    private List<Candidate> searchLayer(List<Double> query, Collection<Node> entries, int ef, int layer, DistanceMetric metric) {
        Comparator<Candidate> ascending = Comparator.comparingDouble(Candidate::distance).thenComparingInt(c -> c.node.item.id());
        PriorityQueue<Candidate> candidates = new PriorityQueue<>(ascending);
        PriorityQueue<Candidate> best = new PriorityQueue<>(ascending.reversed());
        Set<Integer> visited = new HashSet<>();
        for (Node entry : entries) {
            Candidate candidate = new Candidate(entry, metric.distance(query, entry.item.embedding()));
            candidates.offer(candidate); best.offer(candidate); visited.add(entry.item.id());
        }
        while (!candidates.isEmpty()) {
            Candidate nearest = candidates.poll();
            Candidate worst = best.peek();
            if (best.size() >= ef && nearest.distance > worst.distance) break;
            if (layer > nearest.node.level) continue;
            for (int neighborId : nearest.node.neighbors.get(layer)) {
                if (!visited.add(neighborId)) continue;
                Node neighbor = nodes.get(neighborId);
                if (neighbor == null) continue;
                Candidate next = new Candidate(neighbor, metric.distance(query, neighbor.item.embedding()));
                if (best.size() < ef || ascending.compare(next, best.peek()) < 0) {
                    candidates.offer(next); best.offer(next);
                    if (best.size() > ef) best.poll();
                }
            }
        }
        return best.stream().sorted(ascending).toList();
    }

    private double distance(VectorItem a, VectorItem b) {
        return buildMetric.distance(a.embedding(), b.embedding());
    }

    private int maxConnections(int layer) { return layer == 0 ? 2 * m : m; }

    private int randomLevel() {
        double mL = 1.0 / Math.log(m);
        return Math.min(MAX_LEVEL, (int) Math.floor(-Math.log(Math.max(1e-12, random.nextDouble())) * mL));
    }
}
