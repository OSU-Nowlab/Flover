#pragma once

#include <cstddef>
#include <vector>
#include <tbb/concurrent_queue.h>
#include <tbb/tbb.h>
#include <thread>
#include <atomic>
#include <limits>
#include <queue>
#include <map>
#include <numeric>
#include <algorithm>
#include <cmath>


struct mem_block{
    int start;
    int end;
    int cost;
};

long long min_cost_to_move_non_zero_elements(std::vector<mem_block> &arr, int &optimal_start) {
    long long total_cost = 0;
    int non_zero_elements = 0;
    for (int i=0;i<arr.size(); ++i) {
        if (arr[i].cost != 0) {
            non_zero_elements++;
            total_cost += arr[i].cost;
        }
    }

    long long min_cost = std::numeric_limits<long long>::max();
    long long cost_in_current_window = 0;

    // Initialize the first window
    for (int i = 0; i < non_zero_elements; ++i) {
        cost_in_current_window += arr[i].cost;
    }
    min_cost = std::min(min_cost, total_cost - cost_in_current_window);
    optimal_start = 0;

    // Slide the window through the array
    for (int i = non_zero_elements; i < arr.size(); ++i) {
        cost_in_current_window += arr[i].cost - arr[i - non_zero_elements].cost;
        long long current_cost = total_cost - cost_in_current_window;
        if (current_cost < min_cost) {
            min_cost = current_cost;
            optimal_start = i - non_zero_elements + 1;
        }
    }

    return min_cost;
}

void print_move_instructions(std::vector<mem_block> &arr, int optimal_start, std::map<int, int>& my_map) {
    int non_zero_elements = 0;
    for (int i=0;i<arr.size(); ++i) {
        if (arr[i].cost != 0) {
            non_zero_elements++;
        }
    }
    int optimal_end = optimal_start + non_zero_elements - 1;

    std::queue<int> available_positions;
    for (int i = optimal_start; i <= optimal_end; ++i) {
        if (arr[i].cost == 0) {
            available_positions.push(i);
        }
    }

    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i].cost != 0 && (i < optimal_start || i > optimal_end)) {
            int target_position = available_positions.front();
            available_positions.pop();
            // std::cout << "Move element " << i << " (cost: " << arr[i].cost << ") to position " << target_position << std::endl;
            my_map[i] = target_position;
        }
    }
}


void shuffler(std::vector<int> &arr, std::map<int, int>& my_map){
    std::vector<mem_block> cost_arr;
    
    std::vector<int> result;
    int count = 0;
    mem_block start_block;
    start_block.start = 0;
    start_block.end = 1;
    if(arr[0] < 0){
        start_block.cost = 0;
    }else{
        start_block.cost = 1;
    }
    cost_arr.push_back(start_block);

    for (int i = 1; i < arr.size(); ++i) {
        if (arr[i] >= 0) {
            mem_block cur_block;
            cur_block.start = i;
            cur_block.end = i+1;
            cur_block.cost = 1;
            cost_arr.push_back(cur_block);
        } else {
            mem_block cur_block;
            cur_block.start = i;
            cur_block.end = i+1;
            cur_block.cost = 0;
            cost_arr.push_back(cur_block);
        }
    }
    // for(int i=0;i<cost_arr.size();++i){
    //     std::cout<<cost_arr[i].start<<" "<<cost_arr[i].end<<" "<<cost_arr[i].cost<<std::endl;
    // }
    int optimal_start;
    long long min_cost = min_cost_to_move_non_zero_elements(cost_arr, optimal_start);
    // std::cout << "Minimum cost to move all non-zero elements together: " << min_cost << std::endl;
    
    print_move_instructions(cost_arr, optimal_start, my_map);
    // for (const auto &pair : my_map) {
    //     std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
    // }
    
}