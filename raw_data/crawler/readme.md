This folder contains the code for crawlering transaction details from the Etherscan website.

The crawler is designed to search for and download the transaction list based on the token address and transaction hash. All token addresses are listed in the `token_category.csv` file. All transaction hashes for the tokens are listed in the `txhash—Noduplicates.csv` file. The hashes are obtained by the Etherscan's API.


`txhash—Noduplicates.csv` can be downloaded via https://drive.google.com/file/d/1L48Kj6UsRoZnKWDLMCBEDpd2wZu4wtFX/view?usp=sharing.


## Graph Label
The NFTGraph can be employed for graph classification, since the whole graph can be reformed as a graph database by the NFT collection address (Token).

Unlike fungible tokens such as Bitcoin or Ether, each NFT asset belongs to a single collection. Therefore, NFTGraph can be partitioned into various subgraphs/communities. Each subgraph consists of edges that represent transactions trading NFTs in the same collection. In other words, edges with the same edge feature of `Token` are grouped into the same subgraph. Labels of subgraphs are the categories of NFT collections, including art, gaming, and music. The following table presents the statistical outcomes for each category, where PFPs denote the images employed as social media avatars or profile pictures.


| Category | Count |
|:-|:-|
Art | 2872
PFPs | 625
Gaming | 101
Memberships | 93
Photography | 81
Music | 17
Virtual-worlds | 9
Sports-collectibles | 3
Others | 7226
||


