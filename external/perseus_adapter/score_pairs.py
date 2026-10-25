"""Thin native-API wrapper, not a new model/head.

Input parquet: uid (string/int), cutoff_time (datetime[ns], naive UTC), item (string).
One cutoff per user per backbone call avoids EventHub's multi-cutoff priority-union issue.
Missing history / catalog items are returned with null score and an explicit status.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probe_runtime import configure_tempdirs
configure_tempdirs()

import polars as pl
import torch
from perseus import api, core, utils


@torch.no_grad()
def score_pairs(pairs: pl.DataFrame, checkpoint: core.checkpoint.Trained,
                artifacts: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if checkpoint.config.task.type not in {'retrieval','ranking'}:
        raise ValueError('This helper supports retrieval and ranking only')
    pairs=pairs.with_row_index('pair_id').with_columns(
        pl.col('uid').cast(pl.String).alias('client_id'),
        pl.col('item').cast(pl.String),
        pl.col('cutoff_time').cast(pl.Datetime('ns')).alias('timestamp'))
    if utils.torch.device.type=='cuda':
        torch.set_autocast_dtype('cuda',torch.bfloat16)
    wanted=artifacts.join(pairs.select('item').unique(),on='item',how='semi')
    embedded=api.embed_artifacts((wanted,),checkpoint)
    checkpoint.head.layer.eval().to(utils.torch.device)
    with torch.autocast(utils.torch.device.type):
        if checkpoint.config.task.type=='retrieval':
            candidate_embeddings=checkpoint.head.layer(None,embedded)
        else:
            candidate_embeddings=checkpoint.head.layer.item_aggregator(embedded[1])
    candidates=pl.DataFrame({'item':embedded[0],
        'embeddings':pl.Series(candidate_embeddings.float().cpu().numpy(),
                               dtype=pl.Array(pl.Float32,checkpoint.config.backbone.dim))})
    predictions=[]
    users=[]
    # Never combine different cutoffs for the same user in an EventHub read.
    for (cutoff,), cutoff_pairs in pairs.group_by('timestamp',maintain_order=True):
        samples=cutoff_pairs.group_by('client_id','timestamp',maintain_order=True).agg(pl.col('item').unique().alias('items'))
        for partition_id,part in api.distribute_samples(samples):
            embeddings=api.make_backbone_embeddings(partition_id,part,checkpoint)
            if embeddings is None or embeddings.is_empty():
                continue
            users.append(part.select('_index','client_id','timestamp').join(embeddings,on='_index').drop('_index'))
            if checkpoint.config.task.type=='retrieval':
                checkpoint.config.inference.predict_kwargs={'k':0,'only_new':False}
            else:
                checkpoint.config.inference.predict_kwargs={}
            pred=api.make_head_predictions(part,embeddings,checkpoint,embedded_artifacts=embedded)
            predictions.append(pred.select('client_id','timestamp','prediction').explode('prediction').unnest('prediction'))
    if predictions:
        scores=pl.concat(predictions)
        result=pairs.join(scores,on=['client_id','timestamp','item'],how='left')
    else:
        result=pairs.with_columns(score=pl.lit(None,dtype=pl.Float32))
    if users:
        user_embeddings=pl.concat(users)
    else:
        user_embeddings=pl.DataFrame(schema={'client_id':pl.String,'timestamp':pl.Datetime('ns'),
            'embeddings':pl.Array(pl.Float32,checkpoint.config.backbone.dim)})
    known=checkpoint.head.known_items.rename({'_item':'item'}).with_columns(_known=pl.lit(True))
    available=wanted.select('item').with_columns(_artifact=pl.lit(True))
    result=(result.join(available,on='item',how='left').join(known,on='item',how='left')
            .join(user_embeddings.select('client_id','timestamp').with_columns(_history=pl.lit(True)),
                  on=['client_id','timestamp'],how='left')
            .with_columns(status=pl.when(pl.col('_history').is_null()).then(pl.lit('no_history'))
                .when(pl.col('_artifact').is_null()).then(pl.lit('missing_artifact'))
                .when(pl.col('_known').is_null() & (not checkpoint.head.allow_unknown_items)).then(pl.lit('unknown_item'))
                .when(pl.col('score').is_null()).then(pl.lit('unscored'))
                .otherwise(pl.lit('ok')))
            .drop('_known','_artifact','_history').sort('pair_id'))
    return result,user_embeddings,candidates


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--pairs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    external=Path(__file__).resolve().parents[1]
    # User requested that all generated artifacts stay inside external/.
    if not args.output.resolve().is_relative_to(external):
        raise ValueError('Output must be inside external/')
    checkpoint=core.checkpoint.Trained.load(args.checkpoint)
    scores,users,items=score_pairs(pl.read_parquet(args.pairs),checkpoint,pl.read_parquet(args.artifacts))
    args.output.mkdir(parents=True,exist_ok=True)
    scores.write_parquet(args.output/'scores.pq')
    users.write_parquet(args.output/'user_embeddings.pq')
    items.write_parquet(args.output/'candidate_embeddings.pq')
    print(scores.select('status').group_by('status').len())

if __name__=='__main__':
    main()
