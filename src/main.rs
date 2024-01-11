#[allow(unused_imports)]
use c_elegans_22::analysis::*;
#[allow(unused_imports)]
use c_elegans_22::genetic_algorithm::*;
#[allow(unused_imports)]
use c_elegans_22::simulation::*;

fn main() {
    genetic_algorithm_biologically_correct();

    let result: Vec<Ga> = read_result();

    for i in result {
        for j in 8..12 {
            println!("{}", i.gene.gene[j])
        }
        println!()
    }
}
