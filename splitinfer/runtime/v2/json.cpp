#include "json.h"
#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace sjson {
static const Value NULLV;
const Value& Value::operator[](const std::string& k) const {
    if (type != Object) return NULLV;
    auto it = obj.find(k); return it == obj.end() ? NULLV : it->second;
}
struct P {
    const std::string& s; size_t i = 0;
    void ws() { while (i < s.size() && isspace((unsigned char)s[i])) i++; }
    [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("json: ") + m + " at " + std::to_string(i)); }
    Value val() {
        ws(); if (i >= s.size()) fail("eof");
        char c = s[i];
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') { Value v; v.type = Value::String; v.str = string(); return v; }
        if (c == 't' && s.compare(i, 4, "true") == 0) { i += 4; Value v; v.type = Value::Bool; v.b = true; return v; }
        if (c == 'f' && s.compare(i, 5, "false") == 0) { i += 5; Value v; v.type = Value::Bool; return v; }
        if (c == 'n' && s.compare(i, 4, "null") == 0) { i += 4; return Value(); }
        if (c == '-' || isdigit((unsigned char)c)) { Value v; v.type = Value::Number; char* e; v.num = strtod(s.c_str() + i, &e); i = (size_t)(e - s.c_str()); return v; }
        if (c == 'I' || c == 'N') { Value v; v.type = Value::Number; v.num = (c == 'I') ? 1e300 : 0; while (i < s.size() && isalpha((unsigned char)s[i])) i++; return v; }
        fail("unexpected char");
    }
    std::string string() {
        if (s[i] != '"') fail("expected string"); i++;
        std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') { i++; char e = s[i++]; switch (e) { case 'n': out += '\n'; break; case 't': out += '\t'; break; case 'r': out += '\r'; break; case 'u': i += 4; out += '?'; break; default: out += e; } }
            else out += s[i++];
        }
        i++; return out;
    }
    Value array() { Value v; v.type = Value::Array; i++; ws(); if (s[i] == ']') { i++; return v; }
        for (;;) { v.arr.push_back(val()); ws(); if (s[i] == ',') { i++; continue; } if (s[i] == ']') { i++; return v; } fail("array"); } }
    Value object() { Value v; v.type = Value::Object; i++; ws(); if (s[i] == '}') { i++; return v; }
        for (;;) { ws(); std::string k = string(); ws(); if (s[i] != ':') fail("colon"); i++; v.obj[k] = val(); ws();
                   if (s[i] == ',') { i++; continue; } if (s[i] == '}') { i++; return v; } fail("object"); } }
};
Value parse(const std::string& text) { P p{text}; return p.val(); }
}
