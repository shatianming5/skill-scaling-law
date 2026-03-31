"""BM25 Skill 检索。

输入 task description，从 Skill 池中检索最相关的 top-k Skill。
"""

import math
import logging
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果。"""
    skill_id: str
    score: float
    content: str


class SkillRetriever:
    """BM25 Skill 检索器。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[dict] = []       # [{"id": ..., "content": ..., "tokens": [...]}]
        self.doc_freqs: Counter = Counter() # token -> num docs containing it
        self.avg_dl: float = 0.0
        self.n_docs: int = 0
        self._indexed = False

    def index(self, skills: list[dict]):
        """建立 BM25 索引。

        Args:
            skills: Skill 列表，每个包含 'id' 和 'content' 字段
        """
        self.corpus = []
        self.doc_freqs = Counter()

        for skill in skills:
            tokens = self._tokenize(skill["content"])
            self.corpus.append({
                "id": skill["id"],
                "content": skill["content"],
                "tokens": tokens,
            })
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.doc_freqs[t] += 1

        self.n_docs = len(self.corpus)
        total_tokens = sum(len(doc["tokens"]) for doc in self.corpus)
        self.avg_dl = total_tokens / self.n_docs if self.n_docs > 0 else 1.0
        self._indexed = True
        logger.info(f"Indexed {self.n_docs} skills, avg_dl={self.avg_dl:.1f}")

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """检索 top-k 最相关的 Skill。"""
        if not self._indexed:
            raise RuntimeError("Must call index() before retrieve()")

        query_tokens = self._tokenize(query)
        scores = []

        for doc in self.corpus:
            score = self._bm25_score(query_tokens, doc["tokens"])
            scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc, score in scores[:top_k]:
            results.append(RetrievalResult(
                skill_id=doc["id"],
                score=score,
                content=doc["content"],
            ))
        return results

    def _bm25_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        """计算单个文档的 BM25 得分。"""
        doc_len = len(doc_tokens)
        tf_map = Counter(doc_tokens)
        score = 0.0

        for qt in query_tokens:
            if qt not in tf_map:
                continue
            tf = tf_map[qt]
            df = self.doc_freqs.get(qt, 0)
            idf = math.log(
                (self.n_docs - df + 0.5) / (df + 0.5) + 1.0
            )
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            )
            score += idf * tf_norm

        return score

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词（小写 + 空格分割）。"""
        return text.lower().split()

    def evaluate_retrieval(
        self,
        queries: list[dict],
        top_k: int = 3,
    ) -> dict[str, float]:
        """评估检索质量。

        Args:
            queries: [{"query": ..., "relevant_skill_ids": [...]}]
            top_k: 检索数

        Returns:
            {"recall_at_k": float, "ndcg_at_k": float}
        """
        recalls = []
        ndcgs = []

        for q in queries:
            results = self.retrieve(q["query"], top_k)
            retrieved_ids = [r.skill_id for r in results]
            relevant_ids = set(q["relevant_skill_ids"])

            # Recall@k
            hits = len(set(retrieved_ids) & relevant_ids)
            recall = hits / len(relevant_ids) if relevant_ids else 0.0
            recalls.append(recall)

            # nDCG@k
            dcg = sum(
                (1.0 if rid in relevant_ids else 0.0) / math.log2(i + 2)
                for i, rid in enumerate(retrieved_ids)
            )
            ideal = sum(
                1.0 / math.log2(i + 2)
                for i in range(min(len(relevant_ids), top_k))
            )
            ndcg = dcg / ideal if ideal > 0 else 0.0
            ndcgs.append(ndcg)

        return {
            "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
            "ndcg_at_k": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        }

    def compute_recall_at_k(
        self,
        queries: list[str],
        ground_truth_relevant_ids: list[list[str]],
        k: int = 3,
    ) -> dict[str, float]:
        """计算 Recall@K 和 nDCG@K，用于报告 BM25 检索质量协变量。

        与 ``evaluate_retrieval`` 不同，本方法接受扁平的查询字符串列表和
        对应的真值相关 ID 列表，方便直接从实验管线调用以控制
        "结果差可能源于检索差而非 Skill 差"这一混淆因素。

        Parameters
        ----------
        queries : list[str]
            待检索的查询文本列表。
        ground_truth_relevant_ids : list[list[str]]
            每条查询对应的真值相关 Skill ID 列表，长度与 *queries* 相同。
        k : int
            检索截断深度，默认 3。

        Returns
        -------
        dict[str, float]
            包含以下键：

            - ``recall_at_k``  -- 所有查询 Recall@K 的宏平均。
            - ``ndcg_at_k``    -- 所有查询 nDCG@K 的宏平均。
            - ``per_query_recall`` -- 逐查询 Recall 列表（便于方差分析）。
            - ``per_query_ndcg``   -- 逐查询 nDCG 列表。

        Raises
        ------
        RuntimeError
            如果尚未调用 ``index()``。
        ValueError
            如果 *queries* 与 *ground_truth_relevant_ids* 长度不一致。
        """
        if not self._indexed:
            raise RuntimeError("Must call index() before compute_recall_at_k()")
        if len(queries) != len(ground_truth_relevant_ids):
            raise ValueError(
                f"queries ({len(queries)}) and ground_truth_relevant_ids "
                f"({len(ground_truth_relevant_ids)}) must have the same length"
            )

        per_query_recall: list[float] = []
        per_query_ndcg: list[float] = []

        for query, relevant_ids_list in zip(queries, ground_truth_relevant_ids):
            results = self.retrieve(query, top_k=k)
            retrieved_ids = [r.skill_id for r in results]
            relevant_set = set(relevant_ids_list)

            # --- Recall@K ---
            hits = len(set(retrieved_ids) & relevant_set)
            recall = hits / len(relevant_set) if relevant_set else 0.0
            per_query_recall.append(recall)

            # --- nDCG@K ---
            dcg = 0.0
            for rank, rid in enumerate(retrieved_ids):
                if rid in relevant_set:
                    dcg += 1.0 / math.log2(rank + 2)  # rank 0 -> log2(2)

            ideal_hits = min(len(relevant_set), k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            per_query_ndcg.append(ndcg)

        n = len(queries)
        mean_recall = sum(per_query_recall) / n if n > 0 else 0.0
        mean_ndcg = sum(per_query_ndcg) / n if n > 0 else 0.0

        logger.info(
            "Retrieval quality (k=%d, n=%d): Recall@K=%.4f  nDCG@K=%.4f",
            k, n, mean_recall, mean_ndcg,
        )

        return {
            "recall_at_k": mean_recall,
            "ndcg_at_k": mean_ndcg,
            "per_query_recall": per_query_recall,
            "per_query_ndcg": per_query_ndcg,
        }
