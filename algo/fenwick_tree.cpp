#include <iostream>
#include <vector>


int low_bit(int x) {
    return x & -x;
}


class FenwickTree {

public:
    int size;
    std::vector<int> tree;
    std::vector<int> data;

    FenwickTree(int size) : size(size) {
        tree.resize(size, 0);
        data.resize(size, 0);
    }

    FenwickTree(const std::vector<int>& data) : size(data.size()), data(data) {
        this->tree = data;
        int data_size = this->size;
        int size = 2, shift = 1;
        while(size <= data_size) {
            for(int idx=size-1;idx<data_size;idx+=size) {
                tree[idx] += tree[idx - shift];
            }
            shift = size;
            size <<= 1;
        }
    }

    FenwickTree(const FenwickTree& other): size(other.size), tree(other.tree), data(other.data) {}

    void update(int index, int value) {
        // change index-th number to value
        if(index <= 0) return; // avoid inf loop when index == 0
        int idx = index - 1;
        int delta = value - data[idx];
        this->data[idx] = value;
        while(index <= this->size) {
            this->tree[index - 1] += delta;
            index += low_bit(index);
        }
    }

    int prefix_sum(int index) const {
        // sum of [0, index)
        int value = 0;
        while(index > 0) {
            value += this->tree[index - 1];
            index = index - low_bit(index);
        }
        return value;
    }

    int range_sum(int left, int right) const {
        // sum of [left, right)
        return prefix_sum(right) - prefix_sum(left);
    }
};


void print_tree(const FenwickTree& tree) {
    std::cout << "Fenwick Tree: " << std::endl;
    std::cout << "Index  Data  Tree" << std::endl;
    for(int i=0;i<tree.size;i++) {
        std::cout << i << "      " << tree.data[i] << "     "<< tree.tree[i] << std::endl;
    }
}

int main() {
    std::vector<int> array = {1, 2, 3, 4, 5};
    FenwickTree fenwick(array);
    print_tree(fenwick);

    std::cout << "Initial sum from prefix 5: " << fenwick.prefix_sum(5) << std::endl;
    fenwick.update(3, 5);
    std::cout << "After update 3rd element to 5" << std::endl;
    print_tree(fenwick);
    fenwick.update(5, 2);
    std::cout << "After update 5th element to 2" << std::endl;
    print_tree(fenwick);
    std::cout << "Sum of [0, 5): " << fenwick.prefix_sum(5) << std::endl;
    std::cout << "Sum of [0, 1): " << fenwick.prefix_sum(1) << std::endl;
    std::cout << "Sum of [1, 4): " << fenwick.range_sum(1, 4) << std::endl;
    return 0;
}
