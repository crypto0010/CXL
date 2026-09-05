/* Minimal JSON reader (no dependencies).  Enough for program.json. */
#ifndef SPLITINFER_V2_JSON_H
#define SPLITINFER_V2_JSON_H
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace sjson {
struct Value {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false; double num = 0; std::string str;
    std::vector<Value> arr; std::map<std::string, Value> obj;

    bool has(const std::string& k) const { return type == Object && obj.count(k); }
    const Value& operator[](const std::string& k) const;
    const Value& operator[](size_t i) const { return arr.at(i); }
    size_t size() const { return type == Array ? arr.size() : obj.size(); }
    double as_num(double d = 0) const { return type == Number ? num : d; }
    long long as_int(long long d = 0) const { return type == Number ? (long long)num : d; }
    bool as_bool(bool d = false) const { return type == Bool ? b : (type == Number ? num != 0 : d); }
    const std::string& as_str() const { return str; }
};
Value parse(const std::string& text);
}
#endif
