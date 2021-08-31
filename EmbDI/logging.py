"""
Copyright 2020 Riccardo CAPPUZZO

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import csv
import os


class params: pass


class metrics: pass


class mem_results: pass


def log_params():
    path = 'pipeline/' + params.par_dict['output_file'] + '.params'
    metrics_dict = {k: v for k, v in metrics.__dict__.items() if not k.startswith('__')}

    if not os.path.exists(path):
        with open(path, 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            header = list(params.par_dict.keys()) + list(metrics_dict.keys()) + list(mem_results.res_dict.keys())
            writer.writerow(header)
            writer.writerow(
                list(params.par_dict.values()) + list(metrics_dict.values()) + list(mem_results.res_dict.values()))
    else:
        with open('pipeline/logging/' + params.par_dict['output_file'] + '.results', 'a') as fp:
            writer = csv.writer(fp, delimiter=',')
            # writer.writerow(list(configuration.keys()))
            writer.writerow(
                list(params.par_dict.values()) + list(metrics_dict.values()) + list(mem_results.res_dict.values()))
