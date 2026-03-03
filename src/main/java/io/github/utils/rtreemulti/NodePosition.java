package io.github.utils.rtreemulti;

import io.github.utils.rtreemulti.geometry.Geometry;

import java.io.Serializable;

//Mutable, not thread-safe
final class NodePosition<T extends Serializable, S extends Geometry> {

    private Node<T, S> node;
    private int position;
    private double priority;

    NodePosition(Node<T, S> node, int position, double priority) {
        this.node = node;
        this.position = position;
        this.priority = priority;
    }


    Node<T, S> node() {
        return node;
    }

    int position() {
        return position;
    }

    boolean hasRemaining() {
        return position != node.count();
    }

    void setPosition(int position) {
        this.position = position;
    }

    double getPriority() {
        return priority;
    }

    @Override
    public String toString() {
        return "NodePosition [position=" + position + ", priority=" + priority + "]";
    }

}
