// see http://herbsutter.com/gotw/_101/

// pimpl_h.h
#ifndef PIMPL_H_H
#define PIMPL_H_H

#include <memory>

template<typename T>
class pimpl {
private:
    std::unique_ptr<T> m;
public:
    pimpl();
    template<typename ...Args> pimpl( Args&& ... );
    ~pimpl();
    T* operator->();
    T& operator*();
};

#endif
