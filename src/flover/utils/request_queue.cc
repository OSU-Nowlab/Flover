#include "request_queue.h"

namespace flover {


int fl_read_start_ids(size_t            batch_size,
                   std::vector<int>* v_start_lengths,
                   std::vector<int>* v_start_ids,
                   size_t&           max_input_len,
                   const int         end_id,
                   const int         beam_width,
                   std::string       file_name)
{
    std::vector<std::vector<int>> tmp_start_ids;
    std::vector<int>              tmp_start_lengths;

    std::ifstream start_id_file(file_name, std::ios::in);
    int           line_num = 0;
    if (start_id_file.is_open()) {
        std::string line;
        while (std::getline(start_id_file, line)) {
            std::stringstream lineStream(line);
            std::string       vals;
            int               i1 = 0;
            std::vector<int>  tmp_vec;
            while (std::getline(lineStream, vals, ',')) {
                tmp_vec.push_back(std::stoi(vals));
                i1++;
            }
            tmp_start_ids.push_back(tmp_vec);
            tmp_start_lengths.push_back(i1);
            line_num++;
        }
        if (batch_size == 0) {
            batch_size = line_num;
        }
    }
    else {
        printf("[WARNING] Cannot open the file '%s'. \n", file_name.c_str());
        max_input_len = 0;
        return 0;
    }

    max_input_len = tmp_start_lengths.data()[0];
    for (uint i = 1; i < (uint)tmp_start_lengths.size(); i++) {
        max_input_len = max_input_len > tmp_start_lengths.data()[i] ? max_input_len : tmp_start_lengths.data()[i];
    }

    while ((int)tmp_start_lengths.size() < batch_size) {
        std::vector<int> padding_ids;
        for (int i = 0; i < max_input_len; i++) {
            padding_ids.push_back(end_id);
        }
        tmp_start_ids.push_back(padding_ids);
        tmp_start_lengths.push_back(max_input_len);
    }

    // Add padding
    for (int i = 0; i < (int)tmp_start_ids.size(); i++) {
        for (int j = (int)tmp_start_ids[i].size(); j < max_input_len; j++) {
            tmp_start_ids[i].push_back(end_id);
        }
    }

    for (int i = 0; i < (int)tmp_start_ids.size(); i++) {
        for (int b = 0; b < beam_width; b++) {
            for (int j = 0; j < (int)tmp_start_ids[i].size(); j++) {
                v_start_ids->push_back(tmp_start_ids[i][j]);
            }
            v_start_lengths->push_back(tmp_start_lengths[i]);
        }
    }
    return batch_size;
}
}