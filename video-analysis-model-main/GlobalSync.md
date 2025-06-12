This part of the data is calculated through the MeetingServiceImpl: computeTeamMeetingSummary() function. It consists of three parts, namely score, hrv, and radar.

```java
public MeetingSummaryVO computeTeamMeetingSummary(List<MeetingTable> meetingTables) {
        MeetingSummaryVO meetingSummaryVO = new MeetingSummaryVO();
    	//Calculate using radar
        radarService.setEngagementAndAgency(meetingTables, meetingSummaryVO);
    	//Calculate using score
        computeAndSetAlignment(meetingTables, meetingSummaryVO);
    	//Calculate using hrv
        computeAndSetStress(meetingTables, meetingSummaryVO);
        computeAndSetBurnout(meetingSummaryVO);
        computeAndSetScore(meetingTables, meetingSummaryVO);
        return meetingSummaryVO;
    }
```

Score and hrv were obtained through CVHandleServiceImpl:handleCV()

```java
public RestResult handleCV(Long meetingID) throws Exception {
	    //get hrv
        Double hrv = CsvUtil.get_hrv(WINDOW_LENGTH_MS, dataR, timeline.get(0) * 1.0, timeline.get(timeline.size() - 1) * 1.0)
                                  .stream()
                                  .filter(num -> num != null && !Double.isNaN(num))
                                  .mapToDouble(Double::doubleValue)
                                  .average()
                                  .orElse(Double.NaN);
    
    ...
        
        //get score
        Score scores = CsvUtil.get_scores(sync_a, sync_v, sync_r);
    
   ...
       
       

}
```

For radar, it needs to be processed twice, using the NLP and CV functions respectively

NlpServiceImpl

```java
public RestResult handleNlp(Long meetingID) throws IOException {
    List<Double> radar_chart_list = new ArrayList<>();
        List<String> r_keys = new ArrayList<>();
        List<Radar> radars = new ArrayList<>();

        NlpUtil.get_radar_components(speakers_time, total_time.get(0), acts_time, emotions_time, sentences_array, radar_chart_list, r_keys, userList);

        LambdaQueryWrapper<MeetingTable> meetingTableLambdaQueryWrapper = new LambdaQueryWrapper<>();
        meetingTableLambdaQueryWrapper.eq(MeetingTable::getMeeting_id, meetingID);
        MeetingTable meetingTable = meetingService.getOne(meetingTableLambdaQueryWrapper);

        for(int i = 0; i < radar_chart_list.size(); i++){
            //At this moment, the score hasn't been calculated yet, and the else branch will be executed.
            if ("Trust and Psychological Safety".equals(r_keys.get(i))
                    && meetingTable != null && meetingTable.getBehavior_score() != null && meetingTable.getBody_score() != null) {
                Double value = radar_chart_list.get(i);
                Double behaviourScore = meetingTable.getBehavior_score();
                Double bodyScore = meetingTable.getBody_score();
                Double rateTrust = behaviourScore / (behaviourScore + bodyScore);
                if (rateTrust > 0.6) rateTrust = 0.6;
                if (rateTrust < 0.4) rateTrust = 0.4;
                Double ratePsy = 1 - rateTrust;
                radars.add(new Radar(meetingID, "Trust", rateTrust * value * 2));
                radars.add(new Radar(meetingID, "Psychological Safety", ratePsy * value * 2));
            } else {
                Radar r = new Radar();
                r.setMeeting_id(meetingID);
                r.setK(r_keys.get(i));
                r.setV(radar_chart_list.get(i));
                radars.add(r);
            }
        }
}
```

CVHandleServiceImpl

```java
public RestResult handleCV(Long meetingID) throws Exception {
    LambdaQueryWrapper<Radar> radarLambdaQueryWrapper = new LambdaQueryWrapper<>();
        radarLambdaQueryWrapper.eq(Radar::getMeeting_id, meetingID);
        List<Radar> radars = radarService.list(radarLambdaQueryWrapper);
        //Trust hasn't been split yet
        if (radars.size() == 5 && scores.getBehavior_score() != null && scores.getBody_score() != null) {
            //1、Find the index of Trust and Psychological Safety
            int ind = -1;
            Double value = 0.0d;
            for (int i = 0; i < radars.size(); i++) {
                if ("Trust and Psychological Safety".equals(radars.get(i).getK())) {
                    ind = i;
                    value = radars.get(i).getV();
                    break;
                }
            }
            if (ind != -1) {
                radars.remove(ind);
                Double behaviourScore = scores.getBehavior_score();
                Double bodyScore = scores.getBody_score();
                Double rateTrust = behaviourScore / (behaviourScore + bodyScore);
                if (rateTrust > 0.6) rateTrust = 0.6;
                if (rateTrust < 0.4) rateTrust = 0.4;
                Double ratePsy = 1 - rateTrust;
                radars.add(new Radar(meetingID, "Trust", rateTrust * value * 2));
                radars.add(new Radar(meetingID, "Psychological Safety", ratePsy * value * 2));
                radarService.removeByMap(deleteMap);
                log.info("[CVHandleServiceImpl][handleCV] meetingID :{}, radars:{}", meetingID, radars);
                Set<Radar> radarSet = new HashSet<>(radars);
                List<Radar> uniqueRadars = new ArrayList<>(radarSet);
                log.info("[CVHandleServiceImpl][handleCV] meetingID :{}, uniqueRadars:{}", meetingID, uniqueRadars);
                radarService.insertRadar(uniqueRadars);
            }
        }
}
```

