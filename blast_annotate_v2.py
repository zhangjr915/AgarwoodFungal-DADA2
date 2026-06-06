#!/usr/bin/env python3
"""
NCBI BLAST-based taxonomic annotation of fungal ITS OTU sequences.
Uses curl for HTTP requests (more reliable than urllib in this environment).
"""

import os
import sys
import time
import re
import subprocess
import xml.etree.ElementTree as ET
import json
import csv

# Paths
FASTA_FILE = "/home/zhhq/.openclaw/workspace-coder/FungalAnalysis/results/top30_otus.fasta"
OUTPUT_CSV = "/home/zhhq/.openclaw/workspace-coder/FungalAnalysis/results/taxonomy.csv"
OUTPUT_REPORT = "/home/zhhq/.openclaw/workspace-coder/FungalAnalysis/results/blast_annotation_report.txt"
CACHE_FILE = "/home/zhhq/.openclaw/workspace-coder/FungalAnalysis/results/blast_cache.json"

BLAST_BASE = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
DELAY_BETWEEN_SUBMISSIONS = 3
POLL_INTERVAL = 20
MAX_POLL_TIME = 600  # 10 min max per query

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def curl_get(url, timeout=60):
    """HTTP GET using curl."""
    try:
        result = subprocess.run(
            ['curl', '-s', '-m', str(timeout), '-A', 'Mozilla/5.0', url],
            capture_output=True, text=True, timeout=timeout+10
        )
        return result.stdout
    except Exception as e:
        print(f"  curl error: {e}")
        return ""

def submit_blast(sequence, hitlist_size=5):
    """Submit a BLAST query via curl and return RID."""
    import urllib.parse
    params = {
        'CMD': 'Put',
        'PROGRAM': 'blastn',
        'DATABASE': 'nt',
        'QUERY': sequence,
        'EXPECT': '1e-10',
        'HITLIST_SIZE': str(hitlist_size),
        'FORMAT_TYPE': 'XML',
    }
    url = BLAST_BASE + "?" + urllib.parse.urlencode(params)
    
    print(f"  Submitting BLAST...")
    html = curl_get(url, timeout=30)
    
    if not html:
        print(f"  ERROR: No response from BLAST server")
        return None
    
    # Extract RID
    rid_match = re.search(r'RID\s*=\s*([A-Z0-9]+)', html)
    if rid_match:
        rid = rid_match.group(1)
        # Also try to get RTOE
        rtoe_match = re.search(r'RTOE\s*=\s*(\d+)', html)
        rtoe = int(rtoe_match.group(1)) if rtoe_match else 30
        print(f"  RID: {rid} (est. {rtoe}s)")
        return rid
    else:
        print(f"  ERROR: Could not extract RID")
        return None

def check_blast_status(rid):
    """Check if BLAST query is complete."""
    url = f"{BLAST_BASE}?CMD=Get&FORMAT_OBJECT=SearchInfo&RID={rid}"
    html = curl_get(url, timeout=30)
    
    if 'Status=WAITING' in html:
        return 'WAITING'
    elif 'Status=FAILED' in html:
        return 'FAILED'
    elif 'Status=UNKNOWN' in html:
        return 'UNKNOWN'
    elif 'Status=READY' in html:
        if 'ThereAreHits=yes' in html:
            return 'READY'
        else:
            return 'NO_HITS'
    return 'UNKNOWN'

def get_blast_results_xml(rid):
    """Retrieve BLAST results as XML."""
    url = f"{BLAST_BASE}?CMD=Get&FORMAT_TYPE=XML&RID={rid}"
    return curl_get(url, timeout=120)

def parse_blast_xml(xml_data):
    """Parse BLAST XML and extract hit information."""
    results = []
    if not xml_data:
        return results
    
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
        return results
    
    for iteration in root.iter('Iteration'):
        for hit in iteration.iter('Hit'):
            hit_info = {
                'hit_id': hit.findtext('Hit_id', ''),
                'hit_def': hit.findtext('Hit_def', ''),
                'hit_accession': hit.findtext('Hit_accession', ''),
            }
            
            for hsp in hit.iter('Hsp'):
                identity = float(hsp.findtext('Hsp_identity', '0'))
                align_len = int(hsp.findtext('Hsp_align_len', '1'))
                hit_info['identity'] = identity
                hit_info['align_len'] = align_len
                hit_info['identity_pct'] = round(identity / align_len * 100, 2) if align_len > 0 else 0
                hit_info['evalue'] = hsp.findtext('Hsp_evalue', '1')
                hit_info['bit_score'] = hsp.findtext('Hsp_bit-score', '0')
                break
            
            results.append(hit_info)
    
    return results

def get_taxonomy_entrez(accession):
    """Get taxonomy from NCBI Entrez for a given accession."""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nucleotide&id={accession}&rettype=gb&retmode=xml"
    xml_data = curl_get(url, timeout=30)
    
    if not xml_data:
        return {}
    
    taxonomy = {}
    try:
        root = ET.fromstring(xml_data)
        for gbseq in root.iter('GBSeq'):
            taxonomy['organism'] = gbseq.findtext('GBSeq_organism', '')
            taxonomy['definition'] = gbseq.findtext('GBSeq_definition', '')
            tax_str = gbseq.findtext('GBSeq_taxonomy', '')
            if tax_str:
                taxonomy['taxonomy_full'] = [t.strip() for t in tax_str.split(';')]
            
            for feature in gbseq.iter('GBFeature'):
                if feature.findtext('GBFeature_key', '') == 'source':
                    for qualifier in feature.iter('GBQualifier'):
                        qname = qualifier.findtext('GBQualifier_name', '')
                        qval = qualifier.findtext('GBQualifier_value', '')
                        if qname == 'organism':
                            taxonomy['organism'] = qval
                        elif qname == 'db_xref' and 'taxon:' in qval:
                            taxonomy['taxon_id'] = qval.replace('taxon:', '')
    except ET.ParseError:
        pass
    
    time.sleep(0.4)
    return taxonomy

def classify_from_blast(hit_results, taxonomy_info):
    """Produce classification from BLAST results."""
    result = {
        'Kingdom': 'Fungi', 'Phylum': 'Unknown', 'Class': 'Unknown',
        'Order': 'Unknown', 'Family': 'Unknown', 'Genus': 'Unknown',
        'Species': 'Unknown', 'Identity': 0.0, 'Blast_hit': 'No hits found'
    }
    
    if not hit_results:
        return result
    
    best = hit_results[0]
    result['Identity'] = best.get('identity_pct', 0.0)
    result['Blast_hit'] = best.get('hit_def', '')[:250]
    
    # Use Entrez taxonomy if available
    tax_full = taxonomy_info.get('taxonomy_full', [])
    organism = taxonomy_info.get('organism', '')
    
    ranks = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family']
    for i, rank in enumerate(ranks):
        if i < len(tax_full):
            result[rank] = tax_full[i]
    
    if organism:
        result['Species'] = organism
    
    # Fallback: parse from hit definition
    if not tax_full:
        hit_def = best.get('hit_def', '')
        clean = re.sub(r'\s*(ITS|internal transcribed spacer).*$', '', hit_def, flags=re.IGNORECASE)
        clean = re.sub(r'\s*(18S|28S|5\.8S)\s+.*$', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\s*gene,.*$', '', clean)
        clean = re.sub(r'\s*partial.*$', '', clean)
        clean = re.sub(r'\s*complete.*$', '', clean)
        clean = clean.strip()
        
        match = re.match(r'([A-Z][a-z]+)\s+([a-z]+)', clean)
        if match:
            result['Genus'] = match.group(1)
            if not organism:
                result['Species'] = f"{match.group(1)} {match.group(2)}"
    
    return result

def main():
    print("=" * 70)
    print("NCBI BLAST Taxonomic Annotation of Fungal ITS OTU Sequences")
    print("=" * 70)
    
    # Read sequences using simple parser (no biopython needed)
    sequences = []
    with open(FASTA_FILE) as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences.append((current_id, ''.join(current_seq)))
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences.append((current_id, ''.join(current_seq)))
    
    print(f"\nLoaded {len(sequences)} OTU sequences")
    
    cache = load_cache()
    print(f"Cache: {len(cache)} entries")
    
    all_results = []
    
    for i, (header, seq_str) in enumerate(sequences):
        otu_id = header.split(';')[0]
        size_match = re.search(r'size=(\d+)', header)
        size = int(size_match.group(1)) if size_match else 0
        
        print(f"\n[{i+1}/{len(sequences)}] {otu_id} (size={size}, {len(seq_str)}bp)")
        
        # Check cache
        if otu_id in cache and cache[otu_id].get('completed'):
            print(f"  CACHE HIT")
            all_results.append(cache[otu_id])
            continue
        
        # Submit BLAST
        rid = submit_blast(seq_str)
        if not rid:
            result = {
                'OTU_ID': otu_id, 'Size': size, 'Kingdom': 'Fungi',
                'Phylum': 'Unknown', 'Class': 'Unknown', 'Order': 'Unknown',
                'Family': 'Unknown', 'Genus': 'Unknown', 'Species': 'Unknown',
                'Identity': 0.0, 'Blast_hit': 'Submission failed', 'completed': True
            }
            cache[otu_id] = result
            save_cache(cache)
            all_results.append(result)
            continue
        
        # Poll
        start = time.time()
        status = 'WAITING'
        while time.time() - start < MAX_POLL_TIME:
            status = check_blast_status(rid)
            elapsed = int(time.time() - start)
            print(f"  [{elapsed}s] Status: {status}")
            
            if status == 'READY':
                break
            elif status in ('FAILED', 'UNKNOWN', 'NO_HITS'):
                break
            
            time.sleep(POLL_INTERVAL)
        
        if status != 'READY':
            print(f"  BLAST {status}")
            result = {
                'OTU_ID': otu_id, 'Size': size, 'Kingdom': 'Fungi',
                'Phylum': 'Unknown', 'Class': 'Unknown', 'Order': 'Unknown',
                'Family': 'Unknown', 'Genus': 'Unknown', 'Species': 'Unknown',
                'Identity': 0.0, 'Blast_hit': f'BLAST {status}', 'completed': True
            }
            cache[otu_id] = result
            save_cache(cache)
            all_results.append(result)
            continue
        
        # Get results
        print(f"  Retrieving XML results...")
        xml_data = get_blast_results_xml(rid)
        hits = parse_blast_xml(xml_data)
        print(f"  {len(hits)} hits")
        
        if hits:
            for h in hits[:3]:
                print(f"    - {h.get('hit_def','')[:80]} ({h.get('identity_pct',0)}%)")
        
        # Get taxonomy from best hit
        taxonomy_info = {}
        if hits:
            acc = hits[0].get('hit_accession', '')
            if acc:
                print(f"  Fetching taxonomy for {acc}...")
                taxonomy_info = get_taxonomy_entrez(acc)
                org = taxonomy_info.get('organism', 'N/A')
                tax_full = taxonomy_info.get('taxonomy_full', [])
                print(f"  Organism: {org}")
                print(f"  Taxonomy: {' > '.join(tax_full)}")
        
        # Classify
        classification = classify_from_blast(hits, taxonomy_info)
        result = {
            'OTU_ID': otu_id, 'Size': size,
            **classification,
            'completed': True
        }
        
        print(f"  => {classification['Phylum']} | {classification['Class']} | {classification['Genus']} | {classification['Species']} ({classification['Identity']}%)")
        
        cache[otu_id] = result
        save_cache(cache)
        all_results.append(result)
        
        time.sleep(DELAY_BETWEEN_SUBMISSIONS)
    
    # Write outputs
    write_csv(all_results, OUTPUT_CSV)
    write_report(all_results, OUTPUT_REPORT)
    
    print(f"\n{'=' * 70}")
    print(f"DONE! {len(all_results)} OTUs annotated")
    print(f"  CSV: {OUTPUT_CSV}")
    print(f"  Report: {OUTPUT_REPORT}")

def write_csv(results, path):
    fields = ['OTU_ID','Size','Kingdom','Phylum','Class','Order','Family','Genus','Species','Identity','Blast_hit']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f"  CSV saved: {path}")

def write_report(results, path):
    with open(path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BLAST-based Taxonomic Annotation Report\n")
        f.write("Fungal ITS OTU Sequences\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total OTUs: {len(results)}\n")
        f.write(f"BLAST: blastn vs nt\n\n")
        
        phyla = {}
        total_reads = 0
        for r in results:
            p = r.get('Phylum', 'Unknown')
            phyla[p] = phyla.get(p, 0) + 1
            total_reads += r.get('Size', 0)
        
        f.write("-" * 60 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 60 + "\n\n")
        f.write(f"Total reads represented: {total_reads}\n\n")
        f.write("Phylum distribution:\n")
        for p, c in sorted(phyla.items(), key=lambda x: -x[1]):
            pct = sum(r.get('Size',0) for r in results if r.get('Phylum')==p)
            f.write(f"  {p}: {c} OTUs ({pct} reads, {pct/total_reads*100:.1f}%)\n")
        
        f.write("\n" + "-" * 60 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-" * 60 + "\n\n")
        
        for r in results:
            f.write(f"OTU: {r.get('OTU_ID')} (size={r.get('Size',0)})\n")
            f.write(f"  Kingdom:  {r.get('Kingdom','')}\n")
            f.write(f"  Phylum:   {r.get('Phylum','')}\n")
            f.write(f"  Class:    {r.get('Class','')}\n")
            f.write(f"  Order:    {r.get('Order','')}\n")
            f.write(f"  Family:   {r.get('Family','')}\n")
            f.write(f"  Genus:    {r.get('Genus','')}\n")
            f.write(f"  Species:  {r.get('Species','')}\n")
            f.write(f"  Identity: {r.get('Identity',0)}%\n")
            f.write(f"  Best hit: {r.get('Blast_hit','')}\n\n")
    
    print(f"  Report saved: {path}")

if __name__ == '__main__':
    main()
