# Kaggle competition "predict-redhat-business-value"

needs(skaggle, data.table, caret, feather, readr, ROCR, FNN, xgboost, zoo)

do.preprocess = T
do.load       = T
do.validate   = F
do.submit     = T

approach = 'xgb' # { interp, xgb, combined }
debug.mode = F
rng.seed = 83792 # NOTE: I used: (123, 1986, 94111, 49323, 64047, 90210, 726, 54160216, 555, 7)
nr.cv.folds = 5
submit.id = '4k'
use.weights.for.duplicates = F
xgb.use.only.small.groups = F

if (.Platform$OS.type == 'windows') {
  tmp.dir = 'C:/TEMP/kaggle/predict-redhat-business-value'
} else {
  tmp.dir = './tmp'
}

prepare.data = function() {
  # Load the data, do some obvious casts, and merge
  
  ppl = fread('input/people.csv')
  for (col in names(ppl)[which(sapply(ppl, is.logical))]) {
    set(ppl, j = col, value = as.numeric(ppl[[col]]))
  }
  ppl[, date := as.IDate(date)]
  setnames(ppl, 1, 'id')
  setnames(ppl, paste0('ppl.', names(ppl)))
  
  # FIXME in general, it seems that in this problem we usually don't really care about directions 
  # in time (prev/next) but they are interchangeable. So maybe I need "short/long time arm" pairs
  # instead.... on the other hand, the target is always computed causally, so idk. Worth trying if
  # I have time.
  
  # Add unique-ppl.id-based features
  ppl[, nr.ppl.in.group_1 := .N, by = ppl.group_1]
  ppl[, earliest.ppl.in.group_1 := as.integer(min(ppl.date)), by = ppl.group_1]
  ppl[, latest.ppl.in.group_1   := as.integer(max(ppl.date)), by = ppl.group_1]
  ppl[, timespan.ppl.in.group_1 := as.integer(max(ppl.date) - min(ppl.date)), by = ppl.group_1]
  ppl[, ppl.date.density.group_1 := timespan.ppl.in.group_1 / nr.ppl.in.group_1]
  setkey(ppl, ppl.group_1, ppl.date)
  ppl[, days.since.prev.ppl.group := c(NA_integer_, as.integer(diff(ppl.date))), by = ppl.group_1]
  ppl[, days.until.next.ppl.group := c(as.integer(diff(ppl.date)), NA_integer_), by = ppl.group_1]
  ppl[, ppl.nru.char_6.group := uniqueN(ppl.char_6), by = ppl.group_1]
  ppl[, ppl.nru.char_7.group := uniqueN(ppl.char_7), by = ppl.group_1]
  ppl[, ppl.nru.char_8.group := uniqueN(ppl.char_8), by = ppl.group_1]
  # TODO: try these histogram features per group
  #ppl[, tmp.ppl.char_6 := as.factor(ppl.char_6)]
  #ppl[, paste0('ppl.char_6.hist.', gsub(' ', '', levels(ppl$tmp.ppl.char_6), fixed = T)) := as.list(table(tmp.ppl.char_6)/ .N), by = ppl.group_1]
  ppl[, min.group.char_38  := min (ppl.char_38) , by = ppl.group_1]
  ppl[, mean.group.char_38 := mean(ppl.char_38) , by = ppl.group_1]
  ppl[, max.group.char_38  := max (ppl.char_38) , by = ppl.group_1]
  ppl[, sd.group.char_38   := sd  (ppl.char_38) , by = ppl.group_1]
  ppl[, nru.group.char_38  := uniqueN(ppl.char_38), by = ppl.group_1]
  
  dtrain = fread('input/act_train.csv')
  dtest  = fread('input/act_test.csv')
  dtest$outcome = NA
  dat = rbind(dtrain, dtest)
  rm(dtrain, dtest)
  names(dat) = c('ppl.id', 'act.id', 'act.date', 'act.category', paste0('act.char_', 1:10), 'target')
  setkeyv(dat, setdiff(names(dat), 'act.id'))
  dat$key = cumsum(!duplicated(dat[, setdiff(names(dat), 'act.id'), with = F])) # many examples are complete duplicates (except the act.id)
  dat[, nr.dups := .N, by = key]
  dat[, act.date := as.IDate(act.date)]
  
  dat = merge(dat, ppl, by = 'ppl.id', all.x = T, sort = F)
  rm(ppl)

  dat[, ppl.id.num := as.numeric(substr(ppl.id, 5, 20))]
 #dat[, act.id.num := as.numeric(substr(act.id, 6, 20))] # looks like it's useless
  dat[, act.type := as.numeric(substr(act.id, 4, 4))]
  dat[, act.date.num := as.numeric(act.date) - as.numeric(min(act.date))]
  dat[, ppl.date.num := as.numeric(ppl.date) - as.numeric(min(ppl.date))]
  dat[, act.date.year  := year(act.date)]
  dat[, act.date.month := month(act.date)]
  dat[, act.date.mday  := mday(act.date)]
  dat[, act.date.wday  := weekdays(act.date)]
  dat[, ppl.date.year  := year(ppl.date)]
  dat[, ppl.date.month := month(ppl.date)]
  dat[, ppl.date.mday  := mday(ppl.date)]
  dat[, ppl.date.wday  := weekdays(ppl.date)]
  dat[, act.ppl.date.delta := as.numeric(act.date - ppl.date)]
  dat[, ppl.group_1.num := as.numeric(substr(ppl.group_1, 7, 20))]
  dat[, act.char_10.num := as.numeric(substr(act.char_10, 6, 20))]
  dat[, ppl.char_1 := as.integer(as.factor(ppl.char_1))] # it's binary

  # Characters to factors    
  cols = names(dat)[sapply(dat, is.character)]
  dat[, (cols) := lapply(.SD, as.factor), .SDcols = cols]
  
  # Add freq coded factors, and pairwise interactions
  cols = setdiff(cols, 'act.id')
  dat[, (paste0('freq.', cols)) := lapply(.SD, freq.encode), .SDcols = cols]
  
  if (0) {
    # TODO: add freq coding for select 2 (and higher?) way interactions of low cardinality 
    # categoricals (or binary ones at least)
    
    # The following adds all interactions, but it's too big
    cols = c(setdiff(cols, 'ppl.id'), 'act.type', 'act.date.year', paste0('ppl.char_', c(1, 10:37)))
    for (i in 1:(length(cols) - 1)) {
      col = cols[i]
      cols.wo.col = cols[(-(1:i))]
      if (col %in% c('ppl.group_1', 'act.char_10')) cols.wo.col = setdiff(cols.wo.col, c('ppl.group_1', 'act.char_10'))
      dat[, (paste0('freq.', col, '.xtab.', cols.wo.col)) := lapply(.SD, function(x) freq.encode(interaction(x, dat[, col, with = F]))), .SDcols = cols.wo.col]
    }
  }

  # Action-based group_1 features
  dat[, act.category.nru.group := uniqueN(act.category), by = ppl.group_1]
  dat[, paste0('act.category.group.hist.', gsub(' ', '', levels(dat$act.category), fixed = T)) := as.list(table(act.category)/ .N), by = ppl.group_1]
  
  # Add people and group_1 time-series features
  setkey(dat, ppl.id, act.date)
  dat[, days.since.prev.act.ppl   := c(NA_integer_, as.integer(diff(act.date))), by = ppl.id]
  dat[, days.until.next.act.ppl   := c(as.integer(diff(act.date)), NA_integer_), by = ppl.id]
  setkey(dat, ppl.group_1, act.date)
  dat[, days.since.prev.act.group := c(NA_integer_, as.integer(diff(act.date))), by = ppl.group_1]
  dat[, days.until.next.act.group := c(as.integer(diff(act.date)), NA_integer_), by = ppl.group_1]

  # TODO: more
  # - number of actions / by categories in last/next/around T time, density relative to "noraml"
  # - ngrams of actions (probably overkill and I won't have time for this anyway)
  # - precentage of "new" users in group
  
  if (1) {
    review.features = function(dat) {
      idxy = !is.na(dat$target)
      data.frame(
        type      = unlist(lapply(dat, function(x) class(x)[1])),
        n.unique  = unlist(lapply(dat, function(x) length(unique(x)))),
        f.missing = unlist(lapply(dat, function(x) mean(is.na(x)))),
        spear.cor = unlist(lapply(dat, function(x) { idx = !is.na(x); if (!(class(x)[1] %in% c('integer', 'numeric'))) return (NA); cor(x[idx & idxy], y = dat$target[idx & idxy], method = 'spearman') }))
      )
    }
    
    features.analysis = as.data.table(review.features(as.data.frame(dat)), keep.rownames = T)[order(abs(spear.cor), decreasing = T)]
    features.analysis[, abs.spear.cor := abs(spear.cor)]
    features.analysis = features.analysis[order(abs.spear.cor, decreasing = T)]
    save(features.analysis, file = 'base-feature-analysis.RData')
    
    cat(date(), 'Targets:\n')
    print(table(dat$target))
    cat('\n')
    cat(date(), 'Features:\n\n')
    print(features.analysis)
    cat('\n')
  }
  
  save(dat, file = paste0(tmp.dir, '/pp-data.RData'))
}

predict.date.interp = function(dat) {
  # Interpolate the (ppl.group_1, act.date) mean target over the act.date line, per ppl.group_1.
  
  test.act.ids = dat[is.na(target), act.id]
  
  # NOTE that this is different from the original leak script in that the grid created contains all
  # group_1 values, not just those that appear in the trainset.
  
  dat = dat[, .(ppl.id, ppl.group_1, target, act.id, act.date)]
  setkey(dat, ppl.group_1, act.date)
  
  act.date.grid.per.group1 = dat[, .(count = .N, mean.target = mean(target, na.rm = T)), by = .(ppl.group_1, act.date)]
  act.date.grid.per.group1 = act.date.grid.per.group1[CJ(unique(dat$ppl.group_1), seq(min(dat$act.date), max(dat$act.date), 'day')), allow.cartesian = T]
  act.date.grid.per.group1[is.na(count), count := 0L]
  
  do.interp.orig = function(x) {
    # - This function will run for each value of ppl.group_1.
    # - The argument is a vector of mean.target, the mean of the target at each possible act.date.
    # - Many of the values will be missing, since we didn't have any training data for those dates.
    # - We will interpolate the closest values on each side linearly (i.e., regardless of calendar day distance).
  
    guess.all   = 0.32 # 0.44
    delta.first = 0.1
    delta.last  = 0.2
    
    if (all(is.na(x))) x[ceiling(length(x)/2)] = guess.all
    
    # Find all non-NA indexes, combine them with outside borders
    borders = c(1, which(!is.na(x)), length(x) + 1)
    
    # Establish forward- and backward-looking indexes
    forward_border  = borders[2: length(borders)     ]
    backward_border = borders[1:(length(borders) - 1)]
    
    # Prepare vectors for filling
    forward_border_x = x[forward_border]
    forward_border_x[length(forward_border_x)] = abs(forward_border_x[length(forward_border_x) - 1] - delta.last) # this makes for something similar to a bayesian mean that brings us closer to the global mean (which is around 0.4)
    backward_border_x = x[backward_border]
    backward_border_x[1] = abs(forward_border_x[1] - delta.first) # same
    
    # Generate fill vectors (forward and backward ZOH)
    forward_x_fill  = rep(forward_border_x , forward_border - backward_border)
    backward_x_fill = rep(backward_border_x, forward_border - backward_border)
  
    # Linear interpolation
    vec = (forward_x_fill + backward_x_fill) / 2
    
    x[is.na(x)] = vec[is.na(x)]
    return(x)
  }

  # NOTE: this was not better than the original public script interpolation
  do.interp.new = function(x) {
    # Interpolate without ignoring the calendar distance
    
    mask = !is.na(x)
    n = sum(mask)
    if (n == 0) return (0.444)
    if (n < 2) return (rep(mean(x[mask]), length(x)))
    #browser()
    #vec = approx(seq_along(x), x, xout = seq_along(x))$y # linear
    #vec = pmin(pmax(spline(seq_along(x), x, xout = seq_along(x))$y, 0), 1) # cubic spline (fmm)
    mask = !is.na(x); vec = knn.reg(as.matrix(which(mask)), as.matrix(seq_along(mask)), x[mask], 2)$pred # knn
    x[is.na(x)] = vec[is.na(x)]
    #x = vec
    return(x)
  }
  
  act.date.grid.per.group1[, target := do.interp.new(mean.target), by = ppl.group_1]
  preds = merge(dat[, .(act.id, ppl.group_1, act.date)], act.date.grid.per.group1, by = c('ppl.group_1', 'act.date'), all.x = T, sort = F)[act.id %in% test.act.ids, .(act.id, target)]

  return (preds)
}

predict.date.interp.orig = function(dat) {
  test.ppl.ids = unique(dat[is.na(target), ppl.id])
  
  minact.date = min(dat$act.date)
  maxact.date = max(dat$act.date)
  alldays = seq(minact.date, maxact.date, 'day')
  allCompaniesAndDays = data.table(expand.grid(unique(dat$ppl.group_1[!dat$ppl.id %in% test.ppl.ids]), alldays))
  colnames(allCompaniesAndDays) = c('ppl.group_1', 'date.p')
  setkey(allCompaniesAndDays, 'ppl.group_1', 'date.p')
  
  # What are values on days where we have data?
  meanbycomdate = dat[!dat$ppl.id %in% test.ppl.ids, mean(target), by = c('ppl.group_1', 'act.date')]
  
  # Add them to full data grid
  allCompaniesAndDays = merge(allCompaniesAndDays, meanbycomdate, by.x = c('ppl.group_1', 'date.p'), by.y = c('ppl.group_1', 'act.date'), all.x = T)
  
  interpolateFun = function(x) {
    # Find all non-NA indexes, combine them with outside borders
    borders = c(1, which(!is.na(x)), length(x) + 1)
    
    # establish forward- and backward-looking indexes
    forward_border = borders[2:length(borders)]
    backward_border = borders[1:(length(borders) - 1)]
    
    # prepare vectors for filling
    forward_border_x = x[forward_border]
    forward_border_x[length(forward_border_x)] = abs(forward_border_x[length(forward_border_x) - 1] - 0.2) 
    backward_border_x = x[backward_border]
    backward_border_x[1] = abs(forward_border_x[1] - 0.1)
    
    # generate fill vectors
    forward_x_fill = rep(forward_border_x, forward_border - backward_border)
    backward_x_fill = rep(backward_border_x, forward_border - backward_border)
    forward_x_fill_2 = rep(forward_border, forward_border - backward_border) - 1:length(forward_x_fill)
    backward_x_fill_2 = 1:length(forward_x_fill) - rep(backward_border, forward_border - backward_border)
    
    # linear interpolation
    vec = (forward_x_fill + backward_x_fill) / 2
    
    x[is.na(x)] = vec[is.na(x)]
    return(x)
  }
  
  allCompaniesAndDays[, filled := interpolateFun(V1), by = 'ppl.group_1']
  res = merge(dat[, .(act.id, ppl.id, ppl.group_1, act.date, target)], allCompaniesAndDays, all.x = T, all.y = F, by.x = c('ppl.group_1', 'act.date'), by.y = c('ppl.group_1', 'date.p'))
  res = res[res$ppl.id %in% test.ppl.ids, .(act.id, target = filled)]
  return (res)
}

predict.xgb = function(dat.arg) {
  #
  # Meta features from simple stacked models 
  #

  dat = copy(dat.arg) # we need to expand this, some will be by reference, and don't want to lose the original
  
  # NOTE: This will lead to the same "misleading predictiveness" problem in stacking, and it will 
  # also fool my CV unless I do it here after holding out the targets of each fold. This issue is
  # confusing though... so I hope doing it this way is enough. Also, sicne I'm using 5-fold CV, my
  # ppl.id holdout rate in each fold is the same as the private/public rate so it should really be
  # representative of testset performance (i.e., my CV score will not be biased). 
  
  # On the other hand, during training these features will look more predictive than they are in the
  # testset, which will hurt generalization. This is because I allow myself to use target data from
  # the same ppl.id as the one I am predicting for. In order to resolve this I need to hold out, for
  # each row, the ppl.id for the row and 1/5 of all other ppl.ids at random. One way to implement 
  # this is to partition the data to 5-folds as in CV (based on ppl.id), and hold out the entire 
  # fold for each row in the fold.
  
  setkey(dat, ppl.group_1, act.date, act.id)
  
  train.ppl.ids = unique(dat[!is.na(target), ppl.id])
  set.seed(rng.seed)
  cv.folds = createFolds(1:length(train.ppl.ids), k = nr.cv.folds)
  cv.folds = lapply(cv.folds, function(idx) train.ppl.ids[idx])

  add.meta.features.in.place = function(mask) {
    # FIXME do I really have to do it this way? with tmp and assignment to the masked subset?
    
    dat[, tmp.trainset.size.group := sum(!is.na(target)), by = ppl.group_1]
    dat[tmp.trainset.size.group > 0, tmp.mean.target.group := mean(target, na.rm = T), by = ppl.group_1]
    dat[, tmp.prev.target.group := na.locf(target, fromLast = F, na.rm = F), by = ppl.group_1]
    dat[, tmp.next.target.group := na.locf(target, fromLast = T, na.rm = F), by = ppl.group_1]
    dat[target == 0, tmp.target0.date := act.date]
    dat[target == 1, tmp.target1.date := act.date]
    dat[, tmp.prev.target0.date.group := na.locf(tmp.target0.date, fromLast = F, na.rm = F), by = ppl.group_1]
    dat[, tmp.next.target0.date.group := na.locf(tmp.target0.date, fromLast = T, na.rm = F), by = ppl.group_1]
    dat[, tmp.prev.target1.date.group := na.locf(tmp.target1.date, fromLast = F, na.rm = F), by = ppl.group_1]
    dat[, tmp.next.target1.date.group := na.locf(tmp.target1.date, fromLast = T, na.rm = F), by = ppl.group_1]
    dat[, tmp.nr.target.switches.group := sum(abs(diff(as.numeric(na.omit(target))))), by = ppl.group_1]
    dat <<- merge(dat, predict.date.interp(dat)[, .(act.id, tmp.date.interp.pred = target)], by = 'act.id', all.x = T, sort = F)

    dat[mask, trainset.size.group := tmp.trainset.size.group]
    dat[mask, mean.target.group := tmp.mean.target.group]
    dat[mask, prev.target.group := tmp.prev.target.group]
    dat[mask, next.target.group := tmp.next.target.group]
    dat[mask, days.from.prev.target0.group := as.integer(act.date - tmp.prev.target0.date.group)]
    dat[mask, days.tooo.next.target0.group := as.integer(tmp.next.target0.date.group - act.date)]
    dat[mask, days.from.prev.target1.group := as.integer(act.date - tmp.prev.target1.date.group)]
    dat[mask, days.tooo.next.target1.group := as.integer(tmp.next.target1.date.group - act.date)]
    dat[mask, days.to.target0.group := pmin(days.from.prev.target0.group, days.tooo.next.target0.group, na.rm = T)]
    dat[mask, days.to.target1.group := pmin(days.from.prev.target1.group, days.tooo.next.target1.group, na.rm = T)]
    dat[mask, days.to.target.ratio.group := (days.to.target1.group + 1) / (days.to.target0.group + 1)]
    dat[mask, nr.target.switches.group := tmp.nr.target.switches.group]
    dat[mask, date.interp.pred := tmp.date.interp.pred]
  
    # FIXME call my yenc routine? or simply take the average? I guess for high cardinals I need a GLMM or something!
    # if so, some of the features probably need to be made into (numeric and then) factors first?
    ftrs.to.yenc = c('ppl.char_4', 'ppl.char_6', 'ppl.char_8', 'ppl.char_38', 'act.date', 'act.category', paste0('act.char_', 1:10), 'ppl.date', paste0('ppl.char_', 3:9), 'act.date.wday', 'ppl.date.wday')
    for (col in ftrs.to.yenc) {
      dat[, tmp.yenc := mean(target, na.rm = T), by = col]
      dat[mask, paste0('yenc.', col) := tmp.yenc]
    }
    dat[, tmp.yenc := NULL]
    
    # TODO categorical X group interaction yenc features
    
    # TODO: more
    # - statistics about trainset-continuous intervals of target 1 and 0
    # - mean target per other categorical variables - guide with feature importance (but note that ppl.id is useless here due to the train/test splitting scheme)
    # - full nearest neighbor models (I guess on hamming distance because almost everything is cagtegorical)

    dat[, names(dat)[grep('^tmp\\.', names(dat))] := NULL]
  }

  # Generate meta-features for the test set
  test.mask = is.na(dat$target)
  add.meta.features.in.place(test.mask)

  # Generate meta-features for the train set
  for (cv.fold.i in 1:nr.cv.folds) {
    fold.mask = (dat$ppl.id %in% cv.folds[[cv.fold.i]])
    fold.targets = dat[fold.mask, target]
    dat[fold.mask, target := NA]
    add.meta.features.in.place(fold.mask)
    dat[fold.mask, target := fold.targets]
  }
  
  if (use.weights.for.duplicates) {
    # It's not really necessary to keep all of the duplicates as separate examples
    # TODO: I haven't really gotten far with this... so for now don't user it
    act.id.to.key.map = dat[, .(act.id, key)] # this will help us provide act.id for the entire testset
    dat = unique(dat, by = 'key')
  }

  # NOTE: from this point on the row order in dat must not change in order to match dat.ancillary
  dat.ancillary = dat[, .(act.id, key, target, ppl.id, nr.dups, weight = nr.dups / sum(as.numeric(nr.dups)), nr.ppl.in.group_1)]
  dat[, c('act.id', 'key', 'target', 'ppl.id', 'act.date', 'ppl.date') := NULL]

  # Reduce the cardinality of the huge categoricals to save memory (hopefully I used them to their
  # fullest already in my engineered features)
  dat[, act.char_10 := as.character(act.char_10)]
  tmp = dat[, .N, by = act.char_10]
  dat[act.char_10 %in% tmp[N < 20, act.char_10], act.char_10 := NA]
  dat[, act.char_10 := as.factor(act.char_10)]
  dat[, ppl.group_1 := as.character(ppl.group_1)]
  dat[nr.ppl.in.group_1 < 5, ppl.group_1 := NA]
  dat[, ppl.group_1 := as.factor(ppl.group_1)]
  
  # Replace factors with their OHE
  if (0) {
    # NOTE: since it's huge, I'm using sparse.model.matrix rather than dummyVars. This does non-full
    # rank expansion, is this a problem?
    options(na.action = 'na.pass')
    dat = sparse.model.matrix(~ . - 1, dat)
    options(na.action = 'na.omit')
  } else {
    # The above still gives an out of memory warning, and so I'm not sure what it manages to do.
    # Using some tricks from the best public script: (you learn something new every day!)

    factor.cols = names(dat)[sapply(dat, is.factor)]
    nonfactor.cols = names(dat)[sapply(dat, function(x) !is.factor(x))]
    
    ridx = as.integer(1:nrow(dat))
    dat.old = dat
    dat = Matrix(data.matrix(dat.old[, nonfactor.cols, with = F]), sparse = T)
    
    gc()
    for (col in factor.cols) {
      cat(date(), 'adding', col, '\n')
      x = as.integer(dat.old[[col]])
      x[is.na(x)] = max(x, na.rm = T) + 1
      dat = cbind(dat, sparseMatrix(ridx, x))
      gc()
    }
    
    rm(dat.old)
    gc()
  }
  
  #
  # XGB
  #
  
  test.mask = is.na(dat.ancillary$target)
  if (use.weights.for.duplicates) {
    dtrain = xgb.DMatrix(dat[!test.mask, ], weight = dat.ancillary[!test.mask, weight], label = dat.ancillary[!test.mask, target])
    dtest  = xgb.DMatrix(dat[ test.mask, ], weight = dat.ancillary[ test.mask, weight])
  } else if (xgb.use.only.small.groups) {
    train.mask = !test.mask & (dat.ancillary$nr.ppl.in.group_1 < 5)
    dtrain = xgb.DMatrix(dat[train.mask, ], label = dat.ancillary[train.mask, target])
    dtest  = xgb.DMatrix(dat[test.mask , ])
  } else {
    dtrain = xgb.DMatrix(dat[!test.mask, ], label = dat.ancillary[!test.mask, target])
    dtest  = xgb.DMatrix(dat[ test.mask, ])
  }

  xgb.params = list(
    objective         = 'binary:logistic', 
    eval_metric       = 'auc',
    booster           = 'gbtree', 
    eta               = 0.05,
   #subsample         = 0.86,
    colsample_bytree  = 0.3, #0.92,
   #colsample_bylevel = 0.9,
    min_child_weight  = 0,
    gamma             = 0.005,
    max_depth         = 11,
    nrounds           = 600,
    print_every_n     = 10
  )

  gc()
  
  #browser()
  if (0) {
    # For tuning
    all.ppl.ids = unique(dat.ancillary[!test.mask, ppl.id])
    set.seed(rng.seed)
    cv.folds = createFolds(1:length(all.ppl.ids), k = nr.cv.folds)
    cv.folds = lapply(cv.folds, function(idx) which(dat.ancillary[!test.mask, ppl.id] %in% all.ppl.ids[idx]))
    cv.res = xgb.cv(data = dtrain, xgb.params, folds = cv.folds, nrounds = xgb.params$nrounds, print.every.n = xgb.params$print_every_n)
    gc()
    # => with original params: 80 rounds is optimal (0.9900)
    # => with colsample_bytree = 0.05, we don't get far (0.9888 at round 240)
    # => with depth 6 we kind of stall at 0.9898 at 130
    # => with subsample = 0.8, colsample_bytree = 0.7, max_depth = 13: 0.989986 at round 90, but could probably do a bit more with more rounds
    # => original params but no gamma no bylevel and more depth get me to 0.9897 at 90
    # => no subsample, 0.3 colsample, no bylevel, and 500 rounds gave me 0.99047 or something like that, seemd I should keep going with more rounds
  }
  
  xgb.fit = xgb.train(data = dtrain, xgb.params, nrounds = xgb.params$nrounds, watchlist = list(train = dtrain), print.every.n = xgb.params$print_every_n)

  if (0) {
    impo = xgb.importance(colnames(dat), model = xgb.fit)
    print(impo[1:50, ])
    save(impo, file = 'xgb-feature-importance.RData')
    gc()
  }
  
  preds = predict(xgb.fit, dtest)
  
  return (cbind(dat.ancillary[test.mask, .(act.id)], target = preds))
}

# Do stuff
# ==================================================================================================

if (do.preprocess) {
  cat(date(), 'Preprocessing\n')
  prepare.data()
} 

if (do.load) {
  cat(date(), 'Loading\n')
  load(paste0(tmp.dir, '/pp-data.RData')) # => dat

  if (debug.mode) {
    cat(date(), 'NOTE: in debug mode => subsampling data!\n')
    set.seed(rng.seed)
    all.ppl.ids = sample(unique(dat$ppl.id), 1e4)
    mask = dat$ppl.id %in% all.ppl.ids
    dat = dat[mask]
  }
}

if (do.validate) {
  cat(date(), 'Cross-validating\n')
  dat.train = dat[!is.na(target)]
  dat.train.targets = dat.train$target
  
  # FIXME: actually, to make CV similar to the train/test split we have to account for the 
  # precentage of "leaky" examples: those with matching ppl.group_1.
  
  all.ppl.ids = unique(dat.train$ppl.id)
  set.seed(rng.seed)
  cv.folds = createFolds(1:length(all.ppl.ids), k = nr.cv.folds)
  cv.folds = lapply(cv.folds, function(idx) all.ppl.ids[idx])
  cv.aucs = rep(NA, nr.cv.folds)
  
  for (cv.fold.i in 1:nr.cv.folds) {
    fold.mask = (dat.train$ppl.id %in% cv.folds[[cv.fold.i]])
    fold.targets = dat.train[fold.mask, .(act.id, key, nr.dups, target)]
    dat.train[fold.mask, target := NA]
    
    if (approach == 'interp') {
      fold.preds = predict.date.interp(dat.train)
    } else if (approach == 'xgb') {
      fold.preds = predict.xgb(dat.train)
      save(fold.preds, file = paste0(tmp.dir, '/xgb-cv-preds-fold', cv.fold.i, '.RData'))
    } else if (approach == 'combined') {
      fold.preds.interp = predict.date.interp.orig(dat.train)
      #fold.preds.xgb = predict.xgb(dat.train)
      load(paste0(tmp.dir, '/xgb-cv-preds-fold', cv.fold.i, '.RData')) # => fold.preds
      fold.preds.xgb = fold.preds
      
      # TODO: improve this... blend according to some sort of confidence per instance or overall 
      # even, or stack (which is what I kind of do alreay in the xgb model!)
      
      fold.preds = merge(fold.preds.interp, fold.preds.xgb, by = 'act.id', all.x = T, sort = F)
      group.ids.only.in.test = setdiff(unique(dat.train$ppl.group_1), unique(dat.train[!is.na(target), ppl.group_1]))
      act.ids.that.interp.guessed = dat.train[ppl.group_1 %in% group.ids.only.in.test, act.id]
      fold.preds[, target := target.x]
      fold.preds[act.id %in% act.ids.that.interp.guessed, target := target.y]
      fold.preds[, c('target.x', 'target.y') := NULL]
    } else {
      stop('wtf')
    }
    
    if (use.weights.for.duplicates) {
      fold.preds = merge(fold.targets, fold.preds, by = 'key', all.x = T, sort = F)
      fold.preds = fold.preds[rep(1:.N, nr.dups)]
    } else {
      fold.preds = merge(fold.targets, fold.preds, by = 'act.id', all.x = T, sort = F)
    }
    
    stopifnot(!any(is.na(fold.preds$target.y)))
    cv.aucs[cv.fold.i] = as.numeric((ROCR::performance(ROCR::prediction(fold.preds$target.y, fold.preds$target.x), 'auc'))@y.values)
    cat(date(), 'Fold', cv.fold.i, 'AUC:', cv.aucs[cv.fold.i], '\n')
    
    dat.train[, target := dat.train.targets]
  }
  
  cat(date(), ' CV AUC: ', mean(cv.aucs), ' (', sd(cv.aucs) / sqrt(nr.cv.folds), ')\n', sep = '')
  # => With approach interp  : 0.9754726 (0.0021552080) => ?? on pub LB, but note that it achieves CV 0.9985517 (0.0001593045) when considering only those ppl.group_1 values with at least one example in the trainset!
  # => With approach combined: 0.9884348 (0.0009614141), and 0.9809212 (0.0024712470) with xgb.use.only.small.groups
  # => With approach xgb     : 
  # => sbmt-2                : 0.9888410 (0.0009631926) => 0.993051 on pub LB
  # => sbmt-3                : 0.9901693 (0.0009688458) => 0.993027 on pub LB
}

if (do.submit) {
  cat(date(), 'Generating submission\n')
  
  if (approach == 'interp') {
    test.preds = predict.date.interp(dat)[, .(activity_id = act.id, outcome = target)]
  } else if (approach == 'xgb') {
    test.preds = predict.xgb(dat)[, .(activity_id = act.id, outcome = target)]
    save(test.preds, file = paste0(tmp.dir, '/xgb-test-preds.RData'))
  } else if (approach == 'combined') {
    test.preds.interp = predict.date.interp(dat)[, .(activity_id = act.id, outcome = target)]
    #test.preds.xgb = predict.xgb(dat)[, .(activity_id = act.id, outcome = target)]
    load(paste0(tmp.dir, '/xgb-test-preds.RData')) # => test.preds
    test.preds.xgb = test.preds
    test.preds = merge(test.preds.interp, test.preds.xgb, by = 'act.id', all.x = T, sort = F)
    group.ids.only.in.test = setdiff(unique(dat$ppl.group_1), unique(dat[!is.na(target), ppl.group_1]))
    act.ids.that.interp.guessed = dat[ppl.group_1 %in% group.ids.only.in.test, act.id]
    test.preds[, target := target.x]
    test.preds[act.id %in% act.ids.that.interp.guessed, target := target.y]
    test.preds[, c('target.x', 'target.y') := NULL]
  } else {
    stop('wtf')
  }
  
  stopifnot(!any(is.na(test.preds$outcome)))
  write_csv(test.preds, paste0('sbmt-', submit.id, '.csv'))
  zip(paste0('sbmt-', submit.id, '.zip'), paste0('sbmt-', submit.id, '.csv'))
  #ref.preds = fread('best-public-2016-09-12.csv')
  ref.preds = fread('sbmt-2.csv') # my best submission so far (as per the pub lb)
  sanity = merge(ref.preds, test.preds, by = 'activity_id')
  plot(outcome.y ~ outcome.x, sanity, pch = '.', main = 'Sanity check')
}

cat(date(), 'Done.\n')
