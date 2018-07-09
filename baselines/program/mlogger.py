'''
Author: Matthew Zhang
July 5, 2018
'''
import os
import os.path as osp
from collections import defaultdict
import json
import sys

class Writer(object):
    def __init__(self):
        return NotImplementedError
    
    def writekvs(self, kvs):
        return NotImplementedError

class HumanOutputFormat(Writer):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, 'wt')
            self.own_file = True
        else:
            assert hasattr(filename_or_file, 'read'), 'expected file or str, got %s'%filename_or_file
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print('WARNING: tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items()):
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        return s[:20] + '...' if len(s) > 23 else s

    def writeseq(self, seq):
        for arg in seq:
            self.file.write(arg)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()
            
class CSVOutputFormat(Writer):
    def __init__(self, filename):
        self.file = open(filename, 'w+t')
        self.keys = []
        self.sep = ','

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = kvs.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(k)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write('\n')

    def close(self):
        self.file.close()

class Logger():
    def __init__(self, dir=os.getcwd(), subdir=None, output_format=['CSV','HOF']):
        if subdir is not None:
            dir = osp.join(dir, subdir)
        iteration = 0
        is_exist = 1
        while(is_exist):
            temp_dir = osp.join(dir, 'iteration-{}'.format(iteration))
            if not osp.exists(temp_dir):
                dir = temp_dir
                is_exist = 0
            iteration += 1
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.vars = defaultdict(float)
        
        self.output_files = [{'CSV':CSVOutputFormat(osp.join(self.dir, 'progress.csv')),
                              'HOF':HumanOutputFormat(sys.stdout)
                                }[of] for of in output_format]

    def logkv(self, key, val2log):
        self.vars[key]=val2log
        
    def dumpkvs(self):
        for of in self.output_files:
            of.writekvs(self.vars)
        self.vars.clear()
    
    def get_dir(self):
        return self.dir
    
def dump_vars(var2log, dirname, file='vars.json'):
    p = osp.join(dirname, file)
    with open(p, "w") as file:
         file.write(json.dumps(var2log))
    