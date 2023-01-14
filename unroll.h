#ifndef UNROLL_H
#define UNROLL_H


#ifndef NDEBUG
#include <cstdlib>
#include <utility>
#endif


namespace tsr {

  template<size_t start, size_t end>
  struct Unroll{
    // loop body calling a class with operator()
    template<template<size_t> class MapFunc, typename... Args>
    constexpr static void map(MapFunc<start> &&mapfunc, Args&&... args) {
      mapfunc(std::forward<Args...>(args...));
      Unroll<start+1, end>::map(std::forward<MapFunc>(mapfunc),
                                std::forward<Args>(args)...);
    }
    // loop body is a function
    template<typename MapFunc, typename... Args>
    constexpr static void map(MapFunc &&mapfunc, Args&&... args) {
      mapfunc(start, args...);
      Unroll<start+1, end>::map(std::forward<MapFunc>(mapfunc),
                                std::forward<Args>(args)...);
    }
    // loop body is an assignment, this can be replaced with
    // constexpr lambda functions since C++17 (earlier C++
    // version does not support constexpr lambda)
#if __cplusplus < 201703L
    template<typename Lhs, typename Rhs>
    constexpr static void assign(Lhs& lhs, const Rhs& rhs) {
      lhs.sequence_ref(start) = rhs.sequence(start);
      Unroll<start+1, end>::assign(lhs, rhs);
    }
#endif
  };


  template<size_t end>
  struct Unroll<end, end>{
    template<template<size_t> class MapFunc, typename... Args>
    constexpr static void map(MapFunc<end> &&mapfunc, Args&&... args) {}
    template<typename MapFunc, typename... Args>
    constexpr static void map(MapFunc &&mapfunc, Args&&... args) {}
#if __cplusplus < 201703L
    template<typename Lhs, typename Rhs>
    constexpr static void assign(Lhs& lhs, const Rhs& rhs) {}
#endif
  };

} // namespace tsr

#endif
