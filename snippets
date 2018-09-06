class LRUCache {
private:
    list<pair<int, int>> used;
    unordered_map<int, list<pair<int, int>>::iterator>cache;
    int cap;
public:
    LRUCache(int capacity) {
        cap = capacity;
    }
    
    int get(int key) {
        if(cache.find(key) == cache.end()) return -1;
        //list<pair<int, int>>::iterator it = cache.find(key);
        used.splice(used.begin(), used, cache[key]);
        return cache[key]->second;
    }
    
    void put(int key, int value) {
        if(cache.find(key) == cache.end()) { // not present in cache
            if(used.size() == cap) {
                cache.erase(used.back().first); 
                used.pop_back();
            }
            used.push_front(make_pair(key, value));
            cache[key] = used.begin();
        }
        else {// present in cache
            used.splice(used.begin(), used, cache[key]);    
            cache[key]->second = value;
        }
    }
};
