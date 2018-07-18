alerts = [
    {
      "status": "firing",
      "labels": {
        "severity": "OK",
        "service": "apache_ssl_error_logs",
        "site": "sng",
        "hostgroup": "all::dallas-tower",
        "instance": "sngcsmht01",
        "env": "production",
        "alert_source": "nagios"
      },
      "endsAt": "0001-01-01T00:00:00Z",
      "generatorURL": "",
      "startsAt": "2018-04-17T00:48:33.290632395-04:00",
      "annotations": {
        "output": "(17) < 173.36.27.158 - - [02/Apr/2010:21:30:52 -0700] 'GET /houston/health/api/stashes/silence/aer01csmapp4/ntp_sync HTTP/1.1' 404 - ",
        "command": "/etc/sensu/plugs/check_log.sh -F /etc/httpd/logs/ssl_error_log -O /tmp/sensu_ssl_access_log -q 'HTTP/1.1' '[4-5][0-9][0-9]' "
      }
    },
    {
      "status": "firing",
      "labels": {
        "severity": "OK",
        "service": "apache_ssl_error_logs",
        "site": "sng",
        "hostgroup": "all::dallas-tower",
        "instance": "fxccsmht01",
        "env": "production",
        "alert_source": "nagios"
      },
      "endsAt": "0001-01-01T00:00:00Z",
      "generatorURL": "",
      "startsAt": "2018-04-17T00:48:20.129789369-04:00",
      "annotations": {
        "output": "(17) < 173.36.27.158 - - [02/Apr/2010:21:30:52 -0700] 'GET /houston/health/api/stashes/silence/aer01csmapp4/ntp_sync HTTP/1.1' 404 - ",
        "command": "/etc/sensu/plugs/check_log.sh -F /etc/httpd/logs/ssl_error_log -O /tmp/sensu_ssl_access_log -q 'HTTP/1.1' '[4-5][0-9][0-9]' "
      }
    }
  ]

for alert in alerts:
            flattened_dict = {}
            item = alert.items()
            for key, value in item:
                if isinstance(alert[key], dict):
                    flattened_dict.update(alert[key].items())
                else:
                    flattened_dict[key]=value

            print(flattened_dict["startsAt"][:9])
