"""Quality audit of success traces; never rewrites the original success score."""
import argparse
from bisect import bisect_right
import csv
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    rows=json.loads((args.batch/'results.json').read_text());records=[]
    for row in rows:
        if not row['success']:continue
        path=Path(row['artifact_dir'])
        events=[json.loads(x) for x in (path/'diagnostic/controller.jsonl').read_text().splitlines()]
        feedback=[x for x in events if x['kind']=='feedback']
        ft=[x['wall_time'] for x in feedback]
        geometry=[json.loads(x) for x in (path/'geometry_shadow.jsonl').read_text().splitlines()]
        geometry=[g for g in geometry if g.get('valid')]
        gt=[g['wall_time_s'] for g in geometry]
        opening=None
        for event in events:
            if event['kind']!='dispatch' or event['state'][6]<.55:continue
            if min(a[6] for a in event['requested'])>=event['state'][6]-.02:continue
            idx=bisect_right(gt,event['wall_time'])-1
            if idx<0:continue
            g=geometry[idx]['geometry']
            if not (.02<g['insertion_depth_m']<.09 and g['tip_radial_m']<.02 and g['axis_angle_deg']<10):continue
            opening=dict(chunk=event['chunk_id'],wall_time=event['wall_time'],gripper_at_dispatch=event['state'][6],
                         first_requested=event['requested'][0][6],last_requested=event['requested'][-1][6],
                         preceding_geometry=geometry[idx])
            break
        timeline=[]
        for geom in geometry:
            g=geom['geometry']
            if g['tip_radial_m']>=.02 or g['insertion_depth_m']<.02:continue
            idx=bisect_right(ft,geom['wall_time_s'])-1
            if idx<0:continue
            f=feedback[idx]
            timeline.append(dict(wall_time_s=geom['wall_time_s'],depth_m=g['insertion_depth_m'],
                                 tip_radial_m=g['tip_radial_m'],axis_deg=g['axis_angle_deg'],
                                 gripper_actual=f['actual'][6],gripper_desired=f['desired'][6],
                                 inserted=g['candidate_inserted'],fully_seated=g['candidate_fully_seated']))
        name=f"{row['eval_label']}_ep{row['episode']:04d}"
        if timeline:
            with (args.output/(name+'.csv')).open('x') as f:
                writer=csv.DictWriter(f,fieldnames=list(timeline[0]));writer.writeheader();writer.writerows(timeline)
        seated=[g for g in geometry if g['geometry']['candidate_fully_seated']]
        records.append(dict(model=row['eval_label'],episode=row['episode'],artifact=str(path),
            original_success=True,opening_block_near_insertion=opening,
            first_fully_seated=seated[0] if seated else None,
            first_seated_after_opening_command=bool(opening and seated and seated[0]['wall_time_s']>opening['wall_time']),
            release_stability_verified=row['geometry_shadow']['release_stability_verified']))
    result=dict(records=records,limitations=[
        'Opening-block detection is a diagnostic relative drop >0.02rad, not a changed control threshold or success rule.',
        'Geometry is sampled approximately every 0.5 wall seconds; feedback is the preceding available sample.',
        'Seating after an opening command does not alone quantify passive fall distance; it prevents claiming maintained-grasp active insertion without further evidence.'])
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps([dict(model=r['model'],episode=r['episode'],opening_depth_m=(r['opening_block_near_insertion']['preceding_geometry']['geometry']['insertion_depth_m'] if r['opening_block_near_insertion'] else None),seated_after_opening=r['first_seated_after_opening_command']) for r in records],indent=2))


if __name__=='__main__':main()
