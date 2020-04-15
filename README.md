# Reversible Graph to Seqeuence model
## refrence model
* Encoder: [Graph2Seq](https://arxiv.org/abs/1804.00823)

* Decoder: [Semi-Supervised Neural Architecture Search](https://arxiv.org/abs/2002.10389)

* Dataset :
  - [Nasbench101 : paper](https://arxiv.org/abs/1902.09635)
  - [Nasbench101 : code](https://github.com/google-research/nasbench)


## Encoder
- Input : Graph (Adjacency matrix, list of operation)
- output :  sequence

For graph G,
1. Create one virtual node called supernode S.\
**We define embedding(encoded sequence) of G as the node embedding of supernode S**\
Add edges from every node in G to Supernode S.
2. Expand G into sequence.(Follow semiNAS paper method)
3. Loop node embedding : aggregate neighbor nodes.(Follow Graph2Seq paper method)
4. Node embedding of supernode S at the last step is an embedding(encoded sequence) of G

## Decoder
- input: sequence
- ouput : Graph (Adjacency matrix, list of operation)