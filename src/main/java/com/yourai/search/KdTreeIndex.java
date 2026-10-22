package com.yourai.search;

import java.util.*;

/**
 * Balanced, cyclic-axis KD-tree. Exact nearest-neighbour search is supported for
 * Euclidean and Manhattan distance. Cosine distance does not have a generally
 * valid split-plane lower bound, so callers must use brute force for cosine.
 */
public final class KdTreeIndex {
    private final int dimensions;
    private final Node root;
    private final int size;

    private static final class Node {
        final VectorItem item;
        final int axis;
        final Node left;
        final Node right;
        Node(VectorItem item, int axis, Node left, Node right) {
            this.item = item; this.axis = axis; this.left = left; this.right = right;
        }
    }

    public KdTreeIndex(Collection<VectorItem> items) {
        List<VectorItem> copy = new ArrayList<>(items);
        this.size = copy.size();
        this.dimensions = copy.isEmpty() ? 0 : copy.get(0).embedding().size();
        for (VectorItem item : copy) {
            if (item.embedding().size() != dimensions) {
                throw new IllegalArgumentException("All indexed vectors must have the same dimension");
            }
        }
        this.root = build(copy, 0);
    }

    public int size() { return size; }

    public List<BruteForceIndex.Neighbor> search(List<Double> query, int k, DistanceMetric metric) {
        if (metric == DistanceMetric.COSINE) {
            throw new IllegalArgumentException("KD-tree exact pruning supports euclidean and manhattan; use brute force for cosine");
        }
        if (size == 0) return List.of();
        if (query == null || query.size() != dimensions) {
            throw new IllegalArgumentException("Query dimension must match indexed vectors");
        }
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        PriorityQueue<BruteForceIndex.Neighbor> best = new PriorityQueue<>(
            Comparator.comparingDouble(BruteForceIndex.Neighbor::distance).reversed()
                .thenComparing(Comparator.comparingInt((BruteForceIndex.Neighbor n) -> n.item().id()).reversed()));
        search(root, query, Math.min(k, size), metric, best);
        return best.stream().sorted(Comparator.comparingDouble(BruteForceIndex.Neighbor::distance)
            .thenComparingInt(n -> n.item().id())).toList();
    }

    private Node build(List<VectorItem> items, int depth) {
        if (items.isEmpty()) return null;
        int axis = depth % dimensions;
        items.sort(Comparator.comparingDouble((VectorItem item) -> item.embedding().get(axis))
            .thenComparingInt(VectorItem::id));
        int mid = items.size() / 2;
        VectorItem pivot = items.get(mid);
        return new Node(pivot, axis,
            build(new ArrayList<>(items.subList(0, mid)), depth + 1),
            build(new ArrayList<>(items.subList(mid + 1, items.size())), depth + 1));
    }

    private void search(Node node, List<Double> query, int k, DistanceMetric metric,
                        PriorityQueue<BruteForceIndex.Neighbor> best) {
        if (node == null || k == 0) return;
        double distance = metric.distance(query, node.item.embedding());
        BruteForceIndex.Neighbor candidate = new BruteForceIndex.Neighbor(node.item, distance);
        if (best.size() < k) best.offer(candidate);
        else if (compare(candidate, best.peek()) < 0) { best.poll(); best.offer(candidate); }

        double delta = query.get(node.axis) - node.item.embedding().get(node.axis);
        Node near = delta <= 0 ? node.left : node.right;
        Node far = delta <= 0 ? node.right : node.left;
        search(near, query, k, metric, best);

        double worst = best.size() < k ? Double.POSITIVE_INFINITY : best.peek().distance();
        // Distance to the splitting hyperplane is a valid lower bound for L2 and L1.
        if (Math.abs(delta) <= worst) search(far, query, k, metric, best);
    }

    private static int compare(BruteForceIndex.Neighbor a, BruteForceIndex.Neighbor b) {
        int d = Double.compare(a.distance(), b.distance());
        return d != 0 ? d : Integer.compare(a.item().id(), b.item().id());
    }
}
