package com.yourai.search;

import org.springframework.stereotype.Component;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

@Component
public class VectorStore {
    public static final int DEMO_DIMS = 16;
    private final Map<Integer, VectorItem> items = new LinkedHashMap<>();
    private final AtomicInteger nextId = new AtomicInteger(1);
    private KdTreeIndex kdTree = new KdTreeIndex(List.of());
    private final EnumMap<DistanceMetric, HnswIndex> hnswIndexes = new EnumMap<>(DistanceMetric.class);

    public VectorStore() {
        loadDemo();
        rebuildIndexes();
    }

    public synchronized VectorItem insert(String metadata, String category, List<Double> vector) {
        validateVector(vector, DEMO_DIMS, "Demo vectors must have exactly 16 dimensions");
        VectorItem item = new VectorItem(nextId.getAndIncrement(), metadata, category, vector);
        items.put(item.id(), item);
        kdTree = new KdTreeIndex(new ArrayList<>(items.values()));
        for (HnswIndex index : hnswIndexes.values()) index.insert(item);
        return item;
    }

    public synchronized boolean delete(int id) {
        boolean removed = items.remove(id) != null;
        if (removed) rebuildIndexes();
        return removed;
    }

    public synchronized List<VectorItem> all() { return new ArrayList<>(items.values()); }
    public synchronized int size() { return items.size(); }

    public synchronized List<Map<String,Object>> search(List<Double> query, int k, DistanceMetric metric, String algorithm) {
        validateVector(query, DEMO_DIMS, "Query must have exactly 16 dimensions");
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        String requested = normalizeAlgo(algorithm);
        long start = System.nanoTime();
        List<BruteForceIndex.Neighbor> neighbors;
        String actual = requested;
        if (requested.equals("bruteforce")) {
            neighbors = BruteForceIndex.search(items.values(), query, k, metric);
        } else if (requested.equals("kdtree") && metric != DistanceMetric.COSINE) {
            neighbors = kdTree.search(query, k, metric);
        } else if (requested.equals("kdtree")) {
            // Cosine distance does not admit a safe KD split-plane pruning bound in general.
            neighbors = BruteForceIndex.search(items.values(), query, k, metric);
            actual = "bruteforce-fallback";
        } else {
            neighbors = hnswIndexes.get(metric).search(query, k, metric);
        }
        long micros = (System.nanoTime() - start) / 1000;
        List<Map<String,Object>> result = neighbors.stream().map(n -> {
            Map<String,Object> row = new LinkedHashMap<>();
            row.put("id", n.item().id()); row.put("metadata", n.item().metadata());
            row.put("category", n.item().category()); row.put("distance", n.distance());
            row.put("embedding", n.item().embedding()); return row;
        }).toList();
        Map<String,Object> response = new LinkedHashMap<>();
        response.put("results", result); response.put("latencyUs", micros);
        response.put("algo", requested); response.put("actualAlgo", actual);
        response.put("metric", metric.name().toLowerCase());
        if (requested.equals("hnsw")) response.put("efSearch", hnswIndexes.get(metric).efSearch());
        return List.of(response);
    }

    public synchronized Map<String,Object> benchmark(List<Double> query, int k, DistanceMetric metric) {
        validateVector(query, DEMO_DIMS, "Query must have exactly 16 dimensions");
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        Map<String,Object> output = new LinkedHashMap<>();
        List<Integer> exactIds = BruteForceIndex.search(items.values(), query, k, metric).stream()
            .map(n -> n.item().id()).toList();
        output.put("bruteforceUs", timedSearch(query, k, metric, "bruteforce"));
        output.put("kdtreeUs", timedSearch(query, k, metric, "kdtree"));
        output.put("hnswUs", timedSearch(query, k, metric, "hnsw"));
        output.put("bruteforceRecallAtK", 1.0);
        output.put("kdtreeRecallAtK", metric == DistanceMetric.COSINE ? null : recall(exactIds,
            search(query, k, metric, "kdtree").get(0).get("results")));
        output.put("hnswRecallAtK", recall(exactIds, search(query, k, metric, "hnsw").get(0).get("results")));
        output.put("itemCount", size()); output.put("k", Math.min(k, size())); output.put("metric", metric.name().toLowerCase());
        output.put("note", "Timings are in-process measurements and are noisy for small datasets. KD-tree uses exact search for Euclidean/Manhattan and brute-force fallback for cosine. HNSW is approximate; recall@k compares its IDs with brute force.");
        return output;
    }

    private long timedSearch(List<Double> query, int k, DistanceMetric metric, String algo) {
        long start = System.nanoTime(); search(query, k, metric, algo);
        return (System.nanoTime() - start) / 1000;
    }

    private static double recall(List<Integer> expected, Object rawResults) {
        if (expected.isEmpty()) return 1.0;
        if (!(rawResults instanceof List<?> rows)) return 0.0;
        Set<Integer> actual = new HashSet<>();
        for (Object row : rows) if (row instanceof Map<?,?> map && map.get("id") instanceof Number id) actual.add(id.intValue());
        long hits = expected.stream().filter(actual::contains).count();
        return (double) hits / expected.size();
    }

    public synchronized Map<String,Object> graphInfo() {
        return hnswIndexes.get(DistanceMetric.COSINE).graphInfo();
    }

    public synchronized Map<String,Object> stats() {
        return Map.of("count", items.size(), "dims", DEMO_DIMS,
            "algorithms", List.of("bruteforce", "kdtree", "hnsw"),
            "metrics", List.of("euclidean", "cosine", "manhattan"),
            "kdTreeSize", kdTree.size(), "hnswGraphNodes", hnswIndexes.get(DistanceMetric.EUCLIDEAN).size());
    }

    private void rebuildIndexes() {
        List<VectorItem> snapshot = new ArrayList<>(items.values());
        kdTree = new KdTreeIndex(snapshot);
        hnswIndexes.clear();
        for (DistanceMetric metric : DistanceMetric.values()) hnswIndexes.put(metric, new HnswIndex(snapshot, metric));
    }

    private static void validateVector(List<Double> vector, int dimensions, String message) {
        if (vector == null || vector.size() != dimensions) throw new IllegalArgumentException(message);
        for (Double value : vector) {
            if (value == null || !Double.isFinite(value)) throw new IllegalArgumentException("Vector values must be finite numbers");
        }
    }

    private String normalizeAlgo(String algo) {
        String value = algo == null ? "hnsw" : algo.trim().toLowerCase(Locale.ROOT);
        if (!Set.of("hnsw", "kdtree", "bruteforce").contains(value)) {
            throw new IllegalArgumentException("algo must be hnsw, kdtree, or bruteforce");
        }
        return value;
    }

    private void loadDemo() {
        addDemo("Linked List: nodes connected by pointers", "cs", 0);
        addDemo("Binary Search Tree: O(log n) search and insert", "cs", 1);
        addDemo("Dynamic Programming: memoization overlapping subproblems", "cs", 2);
        addDemo("Graph BFS and DFS: breadth and depth first traversal", "cs", 3);
        addDemo("Hash Table: O(1) lookup with collision chaining", "cs", 4);
        addDemo("Calculus: derivatives integrals and limits", "math", 5);
        addDemo("Linear Algebra: matrices eigenvalues eigenvectors", "math", 6);
        addDemo("Probability: distributions random variables Bayes theorem", "math", 7);
        addDemo("Number Theory: primes modular arithmetic RSA cryptography", "math", 8);
        addDemo("Combinatorics: permutations combinations generating functions", "math", 9);
        addDemo("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", 10);
        addDemo("Sushi: vinegared rice raw fish and nori rolls", "food", 11);
        addDemo("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food", 12);
        addDemo("Tacos: corn tortillas with carnitas salsa and cilantro", "food", 13);
        addDemo("Croissant: laminated pastry with buttery flaky layers", "food", 14);
        addDemo("Basketball: fast-paced shooting dribbling slam dunks", "sports", 15);
        addDemo("Football: tackles touchdowns field goals and strategy", "sports", 16);
        addDemo("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports", 17);
        addDemo("Chess: openings endgames tactics strategic board game", "sports", 18);
        addDemo("Swimming: butterfly freestyle backstroke Olympic competition", "sports", 19);
    }

    private void addDemo(String text, String category, int group) {
        List<Double> vector = new ArrayList<>();
        for (int i = 0; i < DEMO_DIMS; i++) {
            double value = i / 4 == group / 5 ? 0.75 + ((i + group) % 4) * 0.04 : 0.05 + ((i + group) % 5) * 0.015;
            vector.add(value);
        }
        VectorItem item = new VectorItem(nextId.getAndIncrement(), text, category, vector);
        items.put(item.id(), item);
    }
}
