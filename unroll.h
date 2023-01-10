#ifndef UNROLL_H
#define UNROLL_H


#ifndef NDEBUG
#include <cstdlib>
#include <utility>
#endif


namespace tsr {

  template <size_t start, size_t end>
  struct Unroll{
    template <template<size_t> class MapFunc, typename... Args>
    static void map(MapFunc<start> &&mapfunc, Args&&... args) {
      mapfunc(std::forward<Args...>(args...));
      Unroll<start+1, end>::map(std::forward<MapFunc>(mapfunc),
                                std::forward<Args>(args)...);
    }
    template <typename MapFunc, typename... Args>
    static void map(MapFunc &&mapfunc, Args&&... args) {
      mapfunc(start, args...);
      Unroll<start+1, end>::map(std::forward<MapFunc>(mapfunc),
                                std::forward<Args>(args)...);
    }
  };


  template <size_t end>
  struct Unroll<end, end>{
    template <template<size_t> class MapFunc, typename... Args>
    static void map(MapFunc<end> &&mapfunc, Args&&... args) {}
    template <typename MapFunc, typename... Args>
    static void map(MapFunc &&mapfunc, Args&&... args) {}
  };

} // namespace tsr
  
#endif
