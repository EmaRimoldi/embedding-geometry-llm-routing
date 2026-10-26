from pathlib import Path
from fractions import Fraction
import itertools, hashlib, json, math, os, sys
import numpy as np

ROOT=Path(__file__).resolve().parent
E4=Path(os.environ['E4_ROOT']); FROZEN=E4/'astra_reward_transfer_20260922_v1/inputs'
sys.path.insert(0,str(ROOT/'vendor'))
from rb.data import load_panel
from rb.panelctx import topic_str

FORMAT={
 'answer':['Return only the final answer, without explanation.','Give the result alone. Omit the working.','State the answer concisely; do not justify it.'],
 'explain':['Explain your reasoning step by step and give the final answer.','Show the working that leads to the result.','Provide a reasoned solution with the intermediate steps.'],
 'python':['Write Python code that computes the answer.','Provide a Python program to solve the problem.','Express the solution as executable Python.'],
 'json':['Return the answer as JSON with the single key "answer".','Produce a JSON object containing only an "answer" field.','Use the JSON format {"answer": your result}.']}

def problem(domain,seed,level):
    rng=np.random.default_rng(1000*domain+seed); a,b,c=rng.integers(2,10,3).tolist()
    if domain==0:
        nums=rng.integers(2,10,2 if level==0 else 10).tolist()
        exp=str(nums[0]); value=nums[0]
        for i,v in enumerate(nums[1:]):
            op='+' if i%2==0 else '*'; exp=f'({exp} {op} {v})'
            value=value+v if op=='+' else value*v
        assert eval(exp,{'__builtins__':{}})==value
        return f'Evaluate the integer expression {exp}.',str(value)
    if domain==1:
        u,v=a+seed,b+2; p,q=2,3
        if level==0: return f'Find x if {a}x + {b} = {a*u+b}.',str(u)
        # Nonzero determinant verified; integer solution is unique.
        assert a*q-b*p != 0 or q==3
        q=3 if a*3-b*2 != 0 else 5
        assert a*q-b*2 != 0
        return f'Solve for x: {a}x + {b}y = {a*u+b*v}, and 2x + {q}y = {2*u+q*v}.',str(u)
    if domain==2:
        modulus=[11,13,17,19][seed%4]; power=3 if level==0 else 101+seed
        return f'What is the remainder when {a}^{power} is divided by {modulus}?',str(pow(a,power,modulus))
    if domain==3:
        n=1 if level==0 else 8; k=1 if level==0 else 3
        ans=math.comb(n,k)*Fraction(a,a+b)**k*Fraction(b,a+b)**(n-k)
        return (f'A bag contains {a} red balls and {b} blue balls. Draw {n} times, replacing the ball after each draw. '
                f'What is the probability of exactly {k} red draws?'),str(ans)
    if domain==4:
        n=8+seed if level==0 else 24+seed; k=2 if level==0 else 9
        return f'A set has {n} distinct elements. How many subsets have exactly {k} elements?',str(math.comb(n,k))
    if domain==5:
        n=2 if level==0 else 12; value=a
        for i in range(n): value=b*value+c
        return f'Let a_0 = {a} and a_(n+1) = {b}a_n + {c}. Find a_{n}.',str(value)
    if domain==6:
        vals=rng.integers(0,2,2 if level==0 else 12).astype(bool).tolist(); exp=str(vals[0]); value=vals[0]
        for i,v in enumerate(vals[1:]):
            op='and' if i%2==0 else 'or'; exp=f'(not {exp} {op} {v})'
            value=((not value) and v) if op=='and' else ((not value) or v)
        assert eval(exp,{'__builtins__':{}})==value
        return f'Evaluate this Boolean expression using Python Boolean semantics: {exp}.',str(value)
    n=2 if level==0 else 14; value=a
    for i in range(n): value=(value*b+i)%97
    return f'What value is printed by this Python code?\nx = {a}\nfor i in range({n}):\n    x = (x * {b} + i) % 97\nprint(x)',str(value)

def main():
    dest=ROOT/'inputs'; dest.mkdir(exist_ok=False)
    sources={}; canaries=[]
    for panel in ('routerbench-0shot','llmrouterbench-performance-cost'):
        p=load_panel(panel); z=np.load(FROZEN/panel/'panel.npz'); ids=z['panel_rows']
        assert np.array_equal(p.query_ids[ids].astype(str),z['query_ids'])
        fam=z['fam']; names=p.datasets
        try: raw=np.asarray(topic_str(p),dtype=str)[ids]
        except Exception:
            # Authoritative topics optional; no LLM-generated-label fallback.
            raw=np.full(len(ids),'',dtype='<U1')
        topic=np.asarray([t if 'mmlu' in names[int(f)].lower() else '' for t,f in zip(raw,fam)])
        fmt=np.full(len(ids),-1,int)
        if panel=='routerbench-0shot':
            fmt=np.asarray([int(names[int(f)] in ('grade-school-math','mbpp','mtbench')) for f in fam])
        # String-only controls, no attempt to infer topic/difficulty from a model.
        texts=p.prompts[ids]
        np.savez_compressed(dest/f'{panel}_metadata.npz',topic=topic,format=fmt,
            chars=np.asarray(list(map(len,texts))),lines=np.asarray([str(t).count('\n')+1 for t in texts]))
        eligible=np.flatnonzero(z['reader_tokens']<512)
        rows=np.random.default_rng(9022).choice(eligible,min(16,len(eligible)),replace=False)
        for row in rows:
            canaries.append(dict(panel=panel,row=int(row),text=str(texts[row]),kind='canary'))
        sources[panel]=json.loads((FROZEN/panel/'manifest.json').read_text())
    corpus=[]; labels=['','This is an introductory exercise.','This is an advanced exercise.']
    for domain,seed,level in itertools.product(range(8),range(12),range(2)):
        core,answer=problem(domain,seed,level); content=domain*12+seed
        for mode,position,sham,terminal in itertools.product(FORMAT,range(2),range(3),range(2)):
            instruction=FORMAT[mode][seed%3]; body=(labels[sham]+'\n'+core).strip()
            text=(instruction+'\n\n'+body) if position==0 else (body+'\n\n'+instruction)
            if terminal: text+='\n\nResponse:'
            corpus.append(dict(kind='controlled',content=content,domain=domain,seed=seed,level=level,
                mode=mode,position=position,sham=sham,terminal=terminal,wording=seed%3,
                verified_answer=answer,text=text))
    for i,row in enumerate(corpus+canaries): row['row']=i if row['kind']=='controlled' else row['row']
    (dest/'controlled.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in corpus+canaries))
    sources['controlled_rows']=len(corpus); sources['canary_rows']=len(canaries)
    sources['controlled_sha256']=hashlib.sha256((dest/'controlled.jsonl').read_bytes()).hexdigest()
    (dest/'sources.json').write_text(json.dumps(sources,indent=2))
    print('prepared controlled',len(corpus),'canaries',len(canaries),flush=True)

if __name__=='__main__':main()
