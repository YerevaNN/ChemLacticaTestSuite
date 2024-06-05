import os

LOGIT_CONFIGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"utils", "logit_configs")
gemma_logit_config_path = os.path.join(LOGIT_CONFIGS_PATH, "gemma_cot.yaml")
galactica_logit_config_path = os.path.join(LOGIT_CONFIGS_PATH, "gal_cot.yaml")

model_125m_122k_9954 = "/auto/home/menuab/code/checkpoints/9954e52e400b43d18d3a40f6/125m_122k_9954"
model_125m_12k_9954 = "/nfs/dgx/raid/chem/checkpoints/facebook/galactica-125m/9954e52e400b43d18d3a40f6/checkpoint-12288"
model_125m_20k_9954 = "/nfs/dgx/raid/chem/checkpoints/facebook/galactica-125m/9954e52e400b43d18d3a40f6/checkpoint-20480"
model_125m_118k_26d3 = "/auto/home/menuab/code/checkpoints/26d322857a184fcbafda5d4a/125m_118k_26d3/"
model_125m_4k_b8cb = "/auto/home/menuab/code/checkpoints/b8cb3a81b61e40aa919e06bc/125m_4k_b8cb/"
model_125m_9k_8073 = "/auto/home/menuab/code/checkpoints/8073deb785f04fcd891e58db/125m_9k_8073/"
model_125m_126k_f3fb = "/auto/home/menuab/code/checkpoints/f3fbd012918247a388efa732/125m_126k_f3fb/"
model_125m_126k_f2c6 = "/auto/home/menuab/code/checkpoints/f2c6ebb289994595a478f513/125m_126k_f2c6/"
model_125m_63k_f2c6 = "/auto/home/menuab/code/checkpoints/f2c6ebb289994595a478f513/125m_63k_f2c6/"
model_125m_313k_cf98 = "/auto/home/menuab/code/checkpoints/cf982665b6c04c83a310b97d/125m_313k_cf98/"
model_125m_512k_fe31 = "/auto/home/menuab/code/checkpoints/fe31d8c5edfd4b93b72f1b60/125m_512k_fe31/"
model_125m_256k_0d99 = "/auto/home/menuab/code/checkpoints/0d992caa5ec443d9aefc289c/125m_256k_0d99/"
model_1b_131k_d5c2   = "/auto/home/menuab/code/checkpoints/d5c2c8db3c554447a27697bf/1.3b_131k_d5c2/"
model_125m_73k_assay_87dc = "/auto/home/menuab/code/checkpoints/87dc7180e49141deae4ded57/125m_73k_assay_87dc/"
model_125m_73k_assay_c6af = "/auto/home/menuab/code/checkpoints/c6af41c79f1244f698cc1153/125m_73k_assay_c6af"
model_125m_18k_a37d = "/nfs/dgx/raid/chem/checkpoints/facebook/galactica-125m/a37d0362e15c4c969307aef8/checkpoint-18432"
model_125m_20k_6913 = "/nfs/dgx/raid/chem/checkpoints/facebook/galactica-125m/6913ba7695b040c597741e76/checkpoint-20480"
model_2b_11k_5292 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/52924785fbfc4c2e839d7e43/checkpoint-11000/"
model_2b_12k_5292 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/52924785fbfc4c2e839d7e43/checkpoint-12000/"
model_2b_20k_869e = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/869e097219da4a4fbbadcc11/checkpoint-20000/"
model_2b_20k_dbf4 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/dbf4eca0f4234b97b2894278/checkpoint-20000/"
model_2b_20k_9283 = "/nfs/ap/mnt/sxtn2/chem/checkpoints/google/gemma-2b/92831d9b0ba24115ad3d2b1e/checkpoint-40000/"
model_2b_20k_2ca2 = "/nfs/ap/mnt/sxtn2/chem/checkpoints/google/gemma-2b/2ca2f355a532478d941ac421/checkpoint-8000/"
# gemma_tokenizer_path = "/auto/home/menuab/code/ChemLactica/chemlactica/tokenizer/GemmaTokenizer"
model_2b_32k_8f45 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/8f4502ae5c354475be62125d/checkpoint-32000/"
model_2b_3k_504e = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/504e413cfff2463db5dd37e4/checkpoint-3200/"
model_2b_16k_a3f8 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/a3f81bf063a14e8289ed0c0c/checkpoint-16000/"
model_2b_1k_d779 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/d779400877344b57b495c8f2/checkpoint-1600/"
model_2b_16k_17b4 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/17b4f40d63ca47dfac3ef6bb/checkpoint-16000/"
model_2b_5k_4b40 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/4b407bb850c744a6a2bbaac5/checkpoint-5000/"

model_2b_4k_6a86 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/6a86c9fd08f040c8838fd8f2/checkpoint-4000"
model_2b_6b_6a86 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/6a86c9fd08f040c8838fd8f2/checkpoint-6000"
model_2b_8k_6a86 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/6a86c9fd08f040c8838fd8f2/checkpoint-8000"
model_2b_10k_6a86 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/6a86c9fd08f040c8838fd8f2/checkpoint-10000"

model_2b_7k_d46c = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/d46cd1cb552847719af4e128/checkpoint-7500"
model_2b_9b_d46c = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/d46cd1cb552847719af4e128/checkpoint-9000"
model_2b_10k_d46c = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/d46cd1cb552847719af4e128/checkpoint-10500"
model_2b_12k_d46c = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/d46cd1cb552847719af4e128/checkpoint-12000"

model_2b_2k_23f7 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/23f7ee79bab74b319447ad28/checkpoint-2000"
model_2b_2k_2995 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/2995fb06c3404625a1e9fbec/checkpoint-2000"
model_2b_2k_4fa8 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/4fa8acfeb35e4b71a41cfe93/checkpoint-2000"

model_125m_20k_13fa = "/nfs/dgx/raid/chem/checkpoints/facebook/galactica-125m/13fa27da6d174d53adfe4c2f/checkpoint-20000"
model_2b_8k_452a = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/452a723110524d27a5dfc438/checkpoint-8000"

model_2b_11k_d6e6 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/d6e6a76e91814ad68d5fa264/checkpoint-11000"
model_2b_2k_699e = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/699e8c6078bb4461a73b39de/checkpoint-2000"

model_125m_18k_1f28 = "/nfs/dgx/raid/chem/checkpoints/facebook/galactica-125m/1f289ff103034364bd27e1c3/checkpoint-18000/"

model_2b_12k_0717 = "/nfs/dgx/raid/chem/checkpoints/h100/google/gemma-2b/0717d445bcf44e31b2887892/checkpoint-12000"
model_2b_18k_0717 = "/nfs/dgx/raid/chem/checkpoints/h100/google/gemma-2b/0717d445bcf44e31b2887892/checkpoint-18000"

gemma_original_tokenizer_path = "google/gemma-2b"
gemma_custom_tokenizer_path = "/auto/home/menuab/code/ChemLactica/chemlactica/tokenizer/GemmaTokenizer"
galactica_tokenizer_path =         "/auto/home/menuab/code/ChemLacticaTestSuite/src/tokenizer/galactica-125m/"
chemlactica_tokenizer_50028_path = "/auto/home/menuab/code/ChemLacticaTestSuite/src/tokenizer/ChemLacticaTokenizer_50028"
chemlactica_tokenizer_50066_path = "/auto/home/menuab/code/ChemLacticaTestSuite/src/tokenizer/ChemLacticaTokenizer_50066"

