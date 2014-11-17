#ifndef QUANTUS_STRUCT_H
#define QUANTUS_STRUCT_H

template <class T>
struct quantus_comm {
    T param1;
    T param2;

    T cache1;
    T cache2;

    int magic_number;

    const T * __restrict matrix;
    size_t pitch;
    int height;
    int width;
    int spacing;
    int offset;
};

#endif
