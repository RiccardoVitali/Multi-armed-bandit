#include <iostream>
#include <random>
#include <vector>

using namespace std;

static double normal_distr(const pair<double,double> & d){
    auto nanosec_since_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::default_random_engine generator(nanosec_since_epoch);
    std::normal_distribution<double> distribution(d.first, d.second);
    return distribution(generator);
}

class bandit{
private:
    const size_t n;
    const vector<pair<int,int>> arm_dist;
    vector<double> rewards;
    vector<int> times;
    int best_arm;
    const double epsilon;
    const int iterations;

public:
    bandit(int n, vector<pair<int,int>> & arm_dist, int iterations, double epsilon) : n(n), arm_dist(arm_dist), iterations(iterations), epsilon(epsilon){}

    int sampling(const int & x){ // 1 = sampling for exploration, 2 = sampling for arm selection
        auto nanosec_since_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::default_random_engine generator(nanosec_since_epoch);
        switch (x) {
            case 1:{
                std::uniform_real_distribution<double> uni_real(0.0, 1.0);
                return uni_real(generator) < epsilon;
            }
            case 2:{
                std::uniform_int_distribution<double> uni_int(0, n - 1);
                return uni_int(generator);
            }
        }
    }


    void update_best(){
        for(size_t i=0;i<n;i++){
            if(i != best_arm && rewards[i] > rewards[best_arm]){
                rewards[best_arm] = rewards[i];
                best_arm = i;
            }
        }
    }

    int eps_greedy_algo(){
        double max = 0;
        for(size_t i=0;i<n;i++){
            double r = normal_distr(arm_dist[i]);
            rewards.emplace_back(r);
            times.emplace_back(1);
            if(r > max){
                best_arm = i;
                max = r;
            }
        }
        double sample;
        for(size_t i=0;i<iterations-n;i++){
            if(sampling(1)){
                size_t selected_arm = sampling(2);
                sample = normal_distr(arm_dist[selected_arm]);
                double temp = rewards[selected_arm];
                rewards[selected_arm] = temp + (sample - temp)/(times[selected_arm]+1);
                times[selected_arm]++;
                if(selected_arm == best_arm && temp > rewards[selected_arm]){
                    update_best();
                }
                if(selected_arm != best_arm && temp < rewards[selected_arm] && rewards[best_arm] < rewards[selected_arm]){
                    best_arm = selected_arm;
                    rewards[best_arm] = rewards[selected_arm];
                }
            }
            else{
                sample = normal_distr(arm_dist[best_arm]);
                double temp = rewards[best_arm];
                rewards[best_arm] = temp + (sample - temp)/times[best_arm];
                times[best_arm]++;
                if(temp > rewards[best_arm]){
                    update_best();
                }
            }
        }
        print_solution();
        return best_arm;
    }

    void print_solution(){
        for(int x=0;x<n;x++){
            cout << "slot " << x << " : " << rewards[x] << " | played " << times[x] << " times" << endl;
        }
        cout << "BEST SLOT: " << best_arm << "\n";
    }
};


int main() {

    vector<pair<int,int>> arm_dist;
    arm_dist.emplace_back(std::make_pair(400,50));
    arm_dist.emplace_back(std::make_pair(405,37));
    arm_dist.emplace_back(std::make_pair(409,35));
    arm_dist.emplace_back(std::make_pair(413,5));
    const int n=arm_dist.size();
    bandit bandito(n,arm_dist,1000,0.1);

    vector<int> sol={0,0,0,0};
    for(int i=0;i<1000;i++){
        auto oo = bandito.eps_greedy_algo();
        sol[oo]++;
    }
    cout << "\nFINAL SOLUTION\n";
    for(int i=0;i<n;i++){
        cout << "slot " << i << " : " << sol[i] << endl;
    }
    return 0;
}