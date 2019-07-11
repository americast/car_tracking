#include <bits/stdc++.h>
#include <fdeep/fdeep.hpp>
int main()
{
    const auto model = fdeep::load_model("fdeep_model.json");
    const auto result = model.predict({
        fdeep::tensor5(fdeep::shape5(1, 1, 224, 224, 3), 42),
        fdeep::tensor5(fdeep::shape5(1, 1, 224, 224, 3), 43)
        });
    std::cout << fdeep::show_tensor5s(result) << std::endl;
}