#= Now, I want to create a bunch of functions which we are going to use in order
to easily estimate excess and redistributed temperature and salinity. This is
going to be based on the MATLAB scripts I wrote to do this, but is going to use
polymorphism in order to be easier to use. =#

using Statistics
using LinearAlgebra
using Distributions, Random
using Base.Threads
using MultivariateStats
include("./DIVA_WrapperFunctions.jl")

function ExcessRedistTempFromTempDIC(initTemp::Vector{Float64},finalTemp::Vector{Float64}
                                    ,initDIC::Vector{Float64},finalDIC::Vector{Float64}
                                    ;transientValue::Float64=0.017
                                    ,gaussianWidthParameter::Float64=0.0001)

    if size(initTemp) == size(finalTemp) == size(initDIC) == size(finalDIC)
      nothing
    else
      error("\"initTemp\", \"finalTemp\", \"initDIC\", \"finalDIC\" must all be the same size")
    end # Could probably wrap this up in a function to make things cleaner?

    VEC_LENGTH::Int64 = length(initTemp)

    ζ = centralDiff(initDIC)
    ξ = centralDiff(initTemp)

    κr = ξ ./ ζ



    weight1 = gaussian(ζ,σ = gaussianWidthParameter)
    weight2 = (weight1 .- 1.0) ./ (κr .- transientValue)

    mat1 = fill(NaN,2,2,VEC_LENGTH)
    mat2 = fill(NaN,2,2,VEC_LENGTH)

    for i = 1:length(κr)
      mat1[:,:,i] = [0               transientValue
                    ;1              -transientValue]
      mat2[:,:,i] = [transientValue -transientValue*κr[i]
                    ;-κr[i]          transientValue*κr[i]]
    end

    ΔΘ = finalTemp - initTemp
    ΔDIC = finalDIC - initDIC

    inputVector = fill(NaN,2,VEC_LENGTH); outputVector = copy(inputVector)
    inputVector[1,:] = ΔΘ
    inputVector[2,:] = ΔDIC

    for i =1:VEC_LENGTH
      outputVector[:,i] = (weight1[i] * mat1[:,:,i] + weight2[i] * mat2[:,:,i]) * inputVector[:,i]
    end

    ExcessTemperature = outputVector[1,:]
    RedistTemperature = outputVector[2,:]

    return ExcessTemperature, RedistTemperature

 end

function ExcessRedistTempFromTempDIC(initTemp::Matrix{Float64},finalTemp::Matrix{Float64}
                                    ,initDIC::Matrix{Float64},finalDIC::Matrix{Float64}
                                    ;transientValue::Float64=0.017
                                    ,zDimension::Int64=1
                                    ,gaussianWidthParameter::Float64=0.0001)

    if size(initTemp) == size(finalTemp) == size(initDIC) == size(finalDIC)
      nothing
    else
      error("\"initTemp\", \"finalTemp\", \"initDIC\", \"finalDIC\" must all be the same size")
    end

    ζ = centralDiff(initDIC,dim=zDimension)
    ξ = centralDiff(initTemp,dim=zDimension)

    κr = ξ ./ ζ

    weight1 = gaussian(ζ,σ = gaussianWidthParameter)
    weight2 = (weight1 .- 1.0) ./ (κr .- transientValue)

    LENGTH_1 = size(initTemp,1)
    LENGTH_2 = size(initTemp,2)

    mat1 = fill(NaN,2,2,LENGTH_1,LENGTH_2)
    mat2 = fill(NaN,2,2,LENGTH_1,LENGTH_2)

    for i = 1:LENGTH_1, j = 1:LENGTH_2
      mat1[:,:,i,j] = [0              transientValue
                    ;1              -transientValue]
      mat2[:,:,i,j] = [transientValue -transientValue*κr[i,j]
                    ;-κr[i,j]         transientValue*κr[i,j] ]
    end

    ΔΘ = finalTemp - initTemp
    ΔDIC = finalDIC - initDIC

    inputVectors = fill(NaN,2,LENGTH_1,LENGTH_2); outputVectors = copy(inputVectors)
    inputVectors[1,:,:] = ΔΘ
    inputVectors[2,:,:] = ΔDIC

    for i =1:LENGTH_1, j = 1:LENGTH_2
      outputVectors[:,i,j] = (weight1[i,j] * mat1[:,:,i,j] + weight2[i,j] * mat2[:,:,i,j]) * inputVectors[:,i,j]
    end

    ExcessTemperature = outputVectors[1,:,:]
    RedistTemperature = outputVectors[2,:,:]

    return ExcessTemperature, RedistTemperature
 end

function ExcessRedistTempFromTempDICHorzSmoothed(initTemp::Matrix{Float64},finalTemp::Matrix{Float64}
                                    ,initDIC::Matrix{Float64},finalDIC::Matrix{Float64}
                                    ;transientValue::Float64=0.017
                                    ,zDimension::Int64=1
                                    ,smoothingWidth::Int64=10
                                    ,gaussianWidthParameter::Float64=0.001)

    if size(initTemp) == size(finalTemp) == size(initDIC) == size(finalDIC)
      nothing
    else
      error("\"initTemp\", \"finalTemp\", \"initDIC\", \"finalDIC\" must all be the same size")
    end

    ζ = centralDiff(initDIC,dim=zDimension)
    ξ = centralDiff(initTemp,dim=zDimension)

    ζ = horzSmoothGradients(ζ,vertDim=zDimension,smoothingWindow=smoothingWidth)
    ξ = horzSmoothGradients(ξ,vertDim=zDimension,smoothingWindow=smoothingWidth)

    κr = ξ ./ ζ

    weight1 = gaussian(ζ,σ = gaussianWidthParameter)
    weight2 = (weight1 .- 1.0) ./ (κr .- transientValue)

    LENGTH_1 = size(initTemp,1)
    LENGTH_2 = size(initTemp,2)

    mat1 = fill(NaN,2,2,LENGTH_1,LENGTH_2)
    mat2 = fill(NaN,2,2,LENGTH_1,LENGTH_2)

    for i = 1:LENGTH_1, j = 1:LENGTH_2
      mat1[:,:,i,j] = [0              transientValue
                    ;1              -transientValue]
      mat2[:,:,i,j] = [transientValue -transientValue*κr[i,j]
                    ;-κr[i,j]         transientValue*κr[i,j] ]
    end

    ΔΘ = finalTemp - initTemp
    ΔDIC = finalDIC - initDIC

    inputMatrix = fill(NaN,2,LENGTH_1,LENGTH_2); outputMatrix = copy(inputMatrix)
    inputMatrix[1,:,:] = ΔΘ
    inputMatrix[2,:,:] = ΔDIC

    for i =1:LENGTH_1 , j = 1:LENGTH_2
      outputMatrix[:,i,j] = (weight1[i,j] * mat1[:,:,i,j] + weight2[i,j] * mat2[:,:,i,j]) * inputMatrix[:,i,j]
    end

    ExcessTemperature = outputMatrix[1,:,:]
    RedistTemperature = outputMatrix[2,:,:]

    return ExcessTemperature, RedistTemperature
 end

function ExcessRedistTempFromTempDIC(Temp::Array{Float64, 3},DIC::Array{Float64, 3}
                                    ;transientValue::Float64=0.017
                                    ,zDimension::Int64=1
                                    ,tDimension::Int64=3
                                    ,gaussianWidthParameter::Float64=0.0001)

    if size(Temp) == size(DIC)
      nothing
    else
      error("\"Temp\" and \"DIC\" must be the same size")
    end

    initTemp = Temp[:,:,1]; initDIC = DIC[:,:,1]
    ExcessTemperature = fill(NaN,size(Temp,1),size(Temp,2),size(Temp,3)-1)
    RedistTemperature = fill(NaN,size(Temp,1),size(Temp,2),size(Temp,3)-1)

    NUM_REPEATS = size(ExcessTemperature,3)

    for i = 1:NUM_REPEATS
      finalTemp = Temp[:,:,i+1]; finalDIC = DIC[:,:,i+1]
      cruiseExcessTemp, cruiseRedistTemp =  ExcessRedistTempFromTempDIC(initTemp,finalTemp
                                    ,initDIC,finalDIC,transientValue=transientValue
                                    ,zDimension=zDimension
                                    ,gaussianWidthParameter=gaussianWidthParameter)

      ExcessTemperature[:,:,i] = cruiseExcessTemp
      RedistTemperature[:,:,i] = cruiseRedistTemp
    end

    return ExcessTemperature, RedistTemperature

 end

function gaussian(x;μ::Float64=0.0,σ::Float64)
  if ndims(x) > 0
    μ = fill(μ,size(x))
    σ = fill(σ,size(x))
  end

  gOfX = exp.( (-1/2) .* ( (x - μ) ./ σ).^2)

  return gOfX
end

function relaxExcess(excessTemp::Matrix{Float64}, redistTemp::Matrix{Float64}
                    ;zDim = 1,scaleFactor::Float64=1.0,sigmoidOffset::Float64=0.5
                    ,numRepeats::Int64=10)

  ΔΘ_init = [excessTemp, redistTemp]
  (ΔΘe_new, ΔΘr_new) = (excessTemp, redistTemp)
  ΔΘ_tot = excessTemp + redistTemp
  ΔΘeScale = excessTempScale(excessTemp,zDimension=zDim)
  for i = 1:numRepeats
    ΔΘe_sym = excessTemp - ΔΘeScale
    ΔΘr_sym = redistTemp - ΔΘeScale
    ΔΘr_new = redistTemp - scaleFactor *ΔΘr_sym .*
     sigmoid.( abs.((excessTemp.*redistTemp).^4 ./ ΔΘ_tot) .- sigmoidOffset )
    ΔΘe_new = excessTemp - scaleFactor *ΔΘe_sym .*
     sigmoid.( abs.((excessTemp.*redistTemp).^4 ./ ΔΘ_tot) .- sigmoidOffset )
    ΔΘ_init = [ΔΘe_new, ΔΘr_new]
    ΔΘeScale = excessTempScale(ΔΘe_new,zDimension=zDim)
  end

  return (ΔΘe_new, ΔΘr_new)
end

function relaxExcess(excessTemp::Array{Float64, 3}, redistTemp::Array{Float64, 3}
                    ;zDim = 1,scaleFactor::Float64=1.0,sigmoidOffset::Float64=0.5
                    ,numRepeats::Int64=10)

  numYears = size(excessTemp,3)
  ΔΘe_new = fill(NaN,size(excessTemp,1),size(excessTemp,2),size(excessTemp,3))
  ΔΘr_new = fill(NaN,size(excessTemp,1),size(excessTemp,2),size(excessTemp,3))
  @threads for i = 1:numYears
    (ΔΘe_new[:,:,i],ΔΘr_new[:,:,i]) = relaxExcess(excessTemp[:,:,i],redistTemp[:,:,i]
                                                ,zDim=zDim,scaleFactor=scaleFactor
                                                ,sigmoidOffset=sigmoidOffset
                                                ,numRepeats=numRepeats)
  end
  return (ΔΘe_new, ΔΘr_new)
end

function excessTempScale(excessTemp::Matrix{Float64};zDimension::Int64=1)

  zDimension != 1  ? excessTemp = excessTemp' : nothing
  ΔθeMedian = fill(NaN,size(excessTemp))

  for i = 1:size(excessTemp,zDimension)
      horzSlice = excessTemp[i,:]
      goodIdx = findNonNaNIndices(horzSlice)
      if length(goodIdx) > 0
          horzSlice = horzSlice[goodIdx]
          ΔθeMedian[i,:] .= median(horzSlice)
      end
  end

  return ΔθeMedian
end

function sigmoid(x)
  value  = 1 / (1 + exp(-x))
  return value
end

function centralDiff(v::AbstractMatrix;dim::Int64=1)
    dvF  = diff(v,dims=dim)
    dvB  = reverse(-diff(reverse(v,dims=dim),dims=dim),dims=dim)
    dx = Matrix{AbstractFloat}(undef,size(v))

    if dim == 1
      dx[1,:] = dvF[1,:]; dx[end,:] = dvB[end,:]
      dx[2:end-1,:] = (dvF[1:end-1,:] + dvB[2:end,:]) / 2
    elseif dim == 2
      dx[:,1] = dvF[:,1]; dx[:,end] = dvB[:,end]
      dx[:,2:end-1] = (dvF[:,1:end-1] + dvB[:,2:end]) / 2
    end

    return dx
end

function ExcessRedistTempSalFromTempSalDIC(initTemp::Matrix{Float64},finalTemp::Matrix{Float64}
                                    ,initSal::Matrix{Float64},finalSal::Matrix{Float64}
                                    ,initDIC::Matrix{Float64},finalDIC::Matrix{Float64}
                                    ;transientValue::Float64=0.017
                                    ,zDimension::Int64=1
                                    ,gaussianWidthParameter::Float64=0.0001)

    if size(initTemp) == size(finalTemp) == size(initDIC) == size(finalDIC) ==
       size(initSal) == size(finalSal)
      nothing
    else
      error("\"initTemp\", \"finalTemp\", \"initDIC\", \"finalDIC\"
            \"initSal\", \"finalSal\" must all be the same size")
    end

    ζ = centralDiff(initDIC,dim=zDimension)
    ξ = centralDiff(initTemp,dim=zDimension)
    η = centralDiff(initSal,dim=zDimension)

    κrT = ξ ./ ζ
    κrS = η ./ ζ
    τ = η ./ ξ

    weight1 = gaussian(ζ,σ = gaussianWidthParameter)
    weight2 = (weight1 .- 1.0) ./ (κrT .- transientValue)

    LENGTH_1 = size(initTemp,1)
    LENGTH_2 = size(initTemp,2)

    mat1 = fill(NaN,3,2,LENGTH_1,LENGTH_2)
    mat2 = fill(NaN,3,2,LENGTH_1,LENGTH_2)

    for i = 1:LENGTH_1, j = 1:LENGTH_2
      mat1[:,:,i,j] = [0              transientValue
                    ;1              -transientValue
                    ;τ[i,j]              -transientValue*τ[i,j]]

      mat2[:,:,i,j] = [transientValue -transientValue*κrT[i,j]
                    ;-κrT[i,j]         transientValue*κrT[i,j]
                    ;-κrS[i,j]         transientValue*κrS[i,j] ]
    end

    ΔΘ = finalTemp - initTemp
    ΔDIC = finalDIC - initDIC
    ΔSal = finalSal - initSal

    inputVectors = fill(NaN,2,LENGTH_1,LENGTH_2);
    outputVectors = fill(NaN,3,LENGTH_1,LENGTH_2);
    inputVectors[1,:,:] = ΔΘ
    inputVectors[2,:,:] = ΔDIC

    for i =1:LENGTH_1, j = 1:LENGTH_2
      outputVectors[:,i,j] = (weight1[i,j] * mat1[:,:,i,j] + weight2[i,j] * mat2[:,:,i,j]) * inputVectors[:,i,j]
    end

    ExcessTemperature = outputVectors[1,:,:]
    RedistTemperature = outputVectors[2,:,:]
    RedistSalinity    = outputVectors[3,:,:]
    ExcessSalinity    = ΔSal - RedistSalinity

    return ExcessTemperature, RedistTemperature, ExcessSalinity, RedistSalinity
 end

function ExcessRedistTempSalFromTempSalDIC(Temp::Array{Float64, 3},Sal::Array{Float64, 3}
                                    ,DIC::Array{Float64, 3}
                                    ;transientValue::Float64=0.017
                                    ,zDimension::Int64=1
                                    ,tDimension::Int64=3
                                    ,gaussianWidthParameter::Float64=0.0001)

    if size(Temp) == size(DIC) == size(Sal)
      nothing
    else
      error("\"Temp\", \"Sal\" and \"DIC\" must be the same size")
    end

    initTemp = Temp[:,:,1]; initDIC = DIC[:,:,1]; initSal = Sal[:,:,1]
    ExcessTemperature = fill(NaN,size(Temp,1),size(Temp,2),size(Temp,3)-1)
    RedistTemperature = fill(NaN,size(Temp,1),size(Temp,2),size(Temp,3)-1)
    ExcessSalinity = fill(NaN,size(Sal,1),size(Sal,2),size(Sal,3)-1)
    RedistSalinity = fill(NaN,size(Sal,1),size(Sal,2),size(Sal,3)-1)

    NUM_REPEATS = size(ExcessTemperature,3)

    for i = 1:NUM_REPEATS
      finalTemp = Temp[:,:,i+1]; finalDIC = DIC[:,:,i+1]; finalSal = Sal[:,:,i+1]
      cruiseExcessTemp, cruiseRedistTemp, cruiseExcessSal, cruiseRedistSal =
      ExcessRedistTempSalFromTempSalDIC(initTemp,finalTemp,initSal,finalSal
                                 ,initDIC,finalDIC,transientValue=transientValue
                                 ,zDimension=zDimension
                                 ,gaussianWidthParameter=gaussianWidthParameter)

      ExcessTemperature[:,:,i] = cruiseExcessTemp
      RedistTemperature[:,:,i] = cruiseRedistTemp
      ExcessSalinity[:,:,i] = cruiseExcessSal
      RedistSalinity[:,:,i] = cruiseRedistSal
    end

    return ExcessTemperature, RedistTemperature, ExcessSalinity, RedistSalinity

 end


function inversionEigenvals(initTemp::Matrix{Float64},initDIC::Matrix{Float64}
                           ;transientValue::Float64=0.017
                           ,zDimension::Int64=1)

    if size(initTemp)  == size(initDIC) 
      nothing
    else
      error("\"initTemp\" and \"initDIC\" must all be the same size")
    end


    ζ = centralDiff(initDIC,dim=zDimension)
    ξ = centralDiff(initTemp,dim=zDimension)

    βT = ξ ./ ζ
    αT = transientValue

    λ1 = (αT .* ( βT .+ 1 ) + sqrt.(Complex.( αT .^2 .* (βT .+ 1) .^2 - 4 .* αT .* βT .* (βT .+ αT)))) ./ (-2 .* (βT .- αT))
    λ2 = (αT .* ( βT .+ 1 ) - sqrt.(Complex.( αT .^2 .* (βT .+ 1) .^2 - 4 .* αT .* βT .* (βT .+ αT)))) ./ (-2 .* (βT .- αT))


    SF1 = min.(1,max.(1 .- log.(abs.(λ1)),0))
    SF2 = min.(1,max.(1 .- log.(abs.(λ2)),0))
    return (λ1 , λ2, SF1, SF2)

end
#  Ideally fuse this and the next function but fuck it for now
function tempDependentInversionEigenvals(initTemp::Matrix{Float64},initDIC::Matrix{Float64}
                                        ;α0::Float64=0.025
                                        ,α1::Float64=-0.0005
                                        ,zDimension::Int64=1)

    if size(initTemp)  == size(initDIC) 
      nothing
    else
      error("\"initTemp\" and \"initDIC\" must all be the same size")
    end


    ζ = centralDiff(initDIC,dim=zDimension)
    ξ = centralDiff(initTemp,dim=zDimension)

    βT = ξ ./ ζ
    αT = initTemp .* α1 .+ α0

    λ1 = (αT .* ( βT .+ 1 ) + sqrt.(Complex.( αT .^2 .* (βT .+ 1) .^2 - 4 .* αT .* βT .* (βT .+ αT)))) ./ (-2 .* (βT .- αT))
    λ2 = (αT .* ( βT .+ 1 ) - sqrt.(Complex.( αT .^2 .* (βT .+ 1) .^2 - 4 .* αT .* βT .* (βT .+ αT)))) ./ (-2 .* (βT .- αT))


    SF1 = min.(1,max.(1 .- log.(abs.(λ1)),0))
    SF2 = min.(1,max.(1 .- log.(abs.(λ2)),0))
    return (λ1 , λ2, SF1, SF2)

end

function inversionEigenvals(Temp::Array{Float64, 3},DIC::Array{Float64,3}
                           ;transientValue::Float64=0.017
                           ,zDimension::Int64=1)

    if size(Temp) == size(DIC)
      nothing
    else
      error("\"Temp\" and \"DIC\" must all be the same size")
    end

    initTemp = Temp[:,:,1]
    initDIC = DIC[:,:,1]

    

    ζ = centralDiff(initDIC,dim=zDimension)
    ξ = centralDiff(initTemp,dim=zDimension)

    βT = ξ ./ ζ
    αT = transientValue

    λ1 = (αT .* ( βT .+ 1 ) + sqrt.(Complex.( αT .^2 .* (βT .+ 1) .^2 - 4 .* αT .* βT .* (βT .+ αT)))) ./ (-2 .* (βT .- αT))
    λ2 = (αT .* ( βT .+ 1 ) - sqrt.(Complex.( αT .^2 .* (βT .+ 1) .^2 - 4 .* αT .* βT .* (βT .+ αT)))) ./ (-2 .* (βT .- αT))


    SF1 = min.(1,max.(1 .- log.(abs.(λ1)),0))
    SF2 = min.(1,max.(1 .- log.(abs.(λ2)),0))
    return (λ1 , λ2, SF1, SF2)

end

mutable struct reinterpolatedFields

  horzCorrLenTe::Vector{Float64}
  vertCorrLenTe::Vector{Float64}

  horzCorrLenTr::Vector{Float64}
  vertCorrLenTr::Vector{Float64}

  horzCorrLenSe::Vector{Float64}
  vertCorrLenSe::Vector{Float64}

  horzCorrLenSr::Vector{Float64}
  vertCorrLenSr::Vector{Float64}

  ExcessTemp::Matrix{Float64}
  RedistTemp::Matrix{Float64}
  ExcessSalt::Matrix{Float64}
  RedistSalt::Matrix{Float64}

  function reinterpolatedFields() # Inner constructor to initialise the
    return(new([0.0]              # reinterpolatedFields struct
              ,[0.0] #
              ,[0.0]
              ,[0.0] #
              ,[0.0]
              ,[0.0] #
              ,[0.0]
              ,[0.0] #
              ,[0.0 0.0; 0.0 0.0]
              ,[0.0 0.0; 0.0 0.0]
              ,[0.0 0.0; 0.0 0.0]
              ,[0.0 0.0; 0.0 0.0]))
  end
end

 function reinterpDecomposedFields(ExcessTemp::Matrix{Float64},RedistTemp::Matrix{Float64}
                                  ,ExcessSal::Matrix{Float64},RedistSal::Matrix{Float64}
                                  ;horzCoordinate::String,meanLatLon::Union{Float64,Int64}
                                  ,horzLenFactor::Number=1,vertLenFactor::Number=1
                                  ,sectionName::String
                                  ,useMask=true
                                  ,maskMatFile::String="/home/ct/Julia/GLODAP_Easy_Ocean/GOSHIP_MaskStruct.mat")
    #=
    This is a function which will take the masked decomposed fields, and
    reinterpolate them to produce a smooth and consistent excess / redistributed
    temperature & salinity field.

    We are going to want to save off the correlation lengths as well as the
    reinterpolated fields, so I'm going to use a struct to keep all of these
    things in one place
    =#

    reinterpolatedFieldsStruct = reinterpolatedFields()

    llGrid, prGrid, mask = loadSectionInfo(MASK_MATFILE,sectionName,gridDir)

    if !useMask
      mask = ones(size(mask))
    end

    (yBoxNum, xBoxNum) = size(ExcessTemp)

    typeof(meanLatLon) == Int64 ? meanLatLon = convert(Float64,meanLatLon) : nothing

    if horzCoordinate == "longitude"
        (lati, loni, prsi) = (fill(meanLatLon,xBoxNum,yBoxNum)
                             ,fill(0.0,xBoxNum,yBoxNum)
                             ,fill(0.0,xBoxNum,yBoxNum) )
        for i = 1:yBoxNum
            loni[:,i] = llGrid
        end
    elseif horzCoordinate == "latitude"
        (lati, loni, prsi) = (fill(0.0,xBoxNum,yBoxNum)
                             ,fill(meanLatLon,xBoxNum,yBoxNum)
                             ,fill(0.0,xBoxNum,yBoxNum) )
        for i = 1:yBoxNum
            lati[:,i] = llGrid
        end
    end

    for i = 1:xBoxNum
        prsi[i,:] = prGrid
    end

    (lati, loni, prsi) = (lati', loni', prsi')
    (lati, loni, prsi) = (lati[:], loni[:], prsi[:])

    (ExcessTemp,RedistTemp) =  (ExcessTemp[:],RedistTemp[:])
    (ExcessSal,RedistSal) =  (ExcessSal[:],RedistSal[:])

    goodIdx = findNonNaNIndices(ExcessTemp)

    lati,loni,prsi = (lati[goodIdx], loni[goodIdx], prsi[goodIdx])

    P_grid,L_grid = ndgrid(prGrid,llGrid)

    horzDist = gridHorzDistance(lati,loni,llGrid)
    vertDist = gridVertDistance(prGrid)
    scaleVert, scaleHorz = calcScaleFactors(vertDist,horzDist)

    zVars = (ExcessTemp[goodIdx],RedistTemp[goodIdx],ExcessSal[goodIdx],RedistSal[goodIdx])

    fieldArray = fill(NaN,length(prGrid),length(llGrid),4)

    for zVar in enumerate(zVars)

      (clVert, clHorz) = calcCorrLengths(zVar[2],obsLat=lati,obsLon=loni
      ,obsPres=prsi,presGrid=prGrid)

      if zVar[1] == 1
        reinterpolatedFieldsStruct.horzCorrLenTe = clHorz
        reinterpolatedFieldsStruct.vertCorrLenTe = clVert
      elseif zVar[1] == 2
        reinterpolatedFieldsStruct.horzCorrLenTr = clHorz
        reinterpolatedFieldsStruct.vertCorrLenTr = clVert
      elseif zVar[1] == 3
        reinterpolatedFieldsStruct.horzCorrLenSe = clHorz
        reinterpolatedFieldsStruct.vertCorrLenSe = clVert
      elseif zVar[1] == 4
        reinterpolatedFieldsStruct.horzCorrLenSr = clHorz
        reinterpolatedFieldsStruct.vertCorrLenSr = clVert
      end
      # There must be a cleaner way of doing this but I dunno what it is

      if horzCoordinate == "longitude"
          reinterpolatedField = easyDIVAGrid(variable=zVar[2],vertVar=prs,latLon=loni
          ,horzCoordinate=horzCoordinate,meanValue="scalar",vertGrid=prGrid,horzGrid=llGrid
          ,horzScale=scaleHorz,vertScale=scaleVert,mask=mask,Epsilon=0.2,horzCorrLength=horzLenFactor*clHorz
          ,vertCorrLength = vertLenFactor*clVert)
        elseif horzCoordinate == "latitude"
          reinterpolatedField = easyDIVAGrid(variable=zVar[2],vertVar=prsi,latLon=lati
          ,horzCoordinate=horzCoordinate,meanValue="scalar",vertGrid=prGrid,horzGrid=llGrid
          ,horzScale=scaleHorz,vertScale=scaleVert,mask=mask,Epsilon=0.2,horzCorrLength=horzLenFactor*clHorz
          ,vertCorrLength = vertLenFactor*clVert)
        end
        fieldArray[:,:,zVar[1]] = reinterpolatedField
      end

      reinterpolatedFieldsStruct.ExcessTemp = fieldArray[:,:,1]
      reinterpolatedFieldsStruct.RedistTemp = fieldArray[:,:,2]
      reinterpolatedFieldsStruct.ExcessSalt = fieldArray[:,:,3]
      reinterpolatedFieldsStruct.RedistSalt = fieldArray[:,:,4]

    return reinterpolatedFieldsStruct

  end

mutable struct reinterpolatedFieldsArray

  horzCorrLenTe::Vector{Float64}
  vertCorrLenTe::Vector{Float64}

  horzCorrLenTr::Vector{Float64}
  vertCorrLenTr::Vector{Float64}

  horzCorrLenSe::Vector{Float64}
  vertCorrLenSe::Vector{Float64}

  horzCorrLenSr::Vector{Float64}
  vertCorrLenSr::Vector{Float64}

  ExcessTemp::Array{Float64, 3}
  RedistTemp::Array{Float64, 3}
  ExcessSalt::Array{Float64, 3}
  RedistSalt::Array{Float64, 3}

  function reinterpolatedFieldsArray(yBoxNum,xBoxNum,tBoxNum) # Inner constructor to initialise the
    return(new([0.0]                                          # reinterpolatedFields struct
              ,[0.0] #
              ,[0.0]
              ,[0.0] #
              ,[0.0]
              ,[0.0] #
              ,[0.0]
              ,[0.0] #
              ,fill(0.0,yBoxNum,xBoxNum,tBoxNum)
              ,fill(0.0,yBoxNum,xBoxNum,tBoxNum)
              ,fill(0.0,yBoxNum,xBoxNum,tBoxNum)
              ,fill(0.0,yBoxNum,xBoxNum,tBoxNum)))
  end
end

 function reinterpDecomposedFields(ExcessTemp::Array{Float64, 3},RedistTemp::Array{Float64, 3}
                                  ,ExcessSal::Array{Float64, 3},RedistSal::Array{Float64, 3}
                                  ;horzCoordinate::String,meanLatLon::Union{Float64,Int64}
                                  ,horzLenFactor::Number=1,vertLenFactor::Number=1
                                  ,sectionName::String
                                  ,maskMatFile::String="/home/ct/Julia/GLODAP_Easy_Ocean/GOSHIP_MaskStruct.mat")
    #=
    This is a function which will take the masked decomposed fields, and
    reinterpolate them to produce a smooth and consistent excess / redistributed
    temperature & salinity field.

    We are going to want to save off the correlation lengths as well as the
    reinterpolated fields, so I'm going to use a struct to keep all of these
    things in one place

    This will work on an array, rather than a matrix. As such, I'm going to set
    it up to only calculate correlation lengths on the last occupation (as we
    expect the signal to be clearest here), as calculating correlation lengths is
    comically slow.
    =#


    llGrid, prGrid, mask = loadSectionInfo(MASK_MATFILE,sectionName,gridDir)

    (yBoxNum, xBoxNum, tBoxNum) = size(ExcessTemp)
    reinterpolatedFieldsStruct = reinterpolatedFieldsArray(yBoxNum, xBoxNum, tBoxNum)

    typeof(meanLatLon) == Int64 ? meanLatLon = convert(Float64,meanLatLon) : nothing

    if horzCoordinate == "longitude"
      (lati, loni, prsi) = (fill(meanLatLon,xBoxNum,yBoxNum), fill(0.0,xBoxNum,yBoxNum), fill(0.0,xBoxNum,yBoxNum) )
      for i = 1:yBoxNum
        loni[:,i] = llGrid
      end
    elseif horzCoordinate == "latitude"
      (lati, loni, prsi) = (fill(0.0,xBoxNum,yBoxNum), fill(meanLatLon,xBoxNum,yBoxNum), fill(0.0,xBoxNum,yBoxNum) )
      for i = 1:yBoxNum
        lati[:,i] = llGrid
      end
    end

    for i = 1:xBoxNum
      prsi[i,:] = prGrid
    end

    (lati, loni, prsi) = (lati', loni', prsi')
    (lati, loni, prsi) = (lati[:], loni[:], prsi[:])


    (ExcessTempVec,RedistTempVec) =  (ExcessTemp[:,:,end][:],RedistTemp[:,:,end][:])
    (ExcessSalVec,RedistSalVec) =  (ExcessSal[:,:,end][:],RedistSal[:,:,end][:])

    goodIdx = findNonNaNIndices(ExcessTempVec)

    nCruisesRemoved = 1
    while length(goodIdx) < 1
      (ExcessTempVec,RedistTempVec) =  (ExcessTemp[:,:,end-nCruisesRemoved][:]
                                       ,RedistTemp[:,:,end-nCruisesRemoved][:])
      (ExcessSalVec,RedistSalVec) =  (ExcessSal[:,:,end-nCruisesRemoved][:]
                                     ,RedistSal[:,:,end-nCruisesRemoved][:])
      goodIdx = findNonNaNIndices(ExcessTempVec)
      nCruisesRemoved += 1
    end

    heatmap()


    lati,loni,prsi = (lati[goodIdx], loni[goodIdx], prsi[goodIdx])

    P_grid,L_grid = ndgrid(prGrid,llGrid)


    horzDist = gridHorzDistance(lati,loni,llGrid)
    vertDist = gridVertDistance(prGrid)
    scaleVert, scaleHorz = calcScaleFactors(vertDist,horzDist)

    # For efficiency, I'm going to remove 2 of every 3 'observations'
    # Has this garbled it? Yes. Since I'm running this over the weekend now, it
    # can be slow
    zVars = (ExcessTempVec[goodIdx],RedistTempVec[goodIdx]
            ,ExcessSalVec[goodIdx],RedistSalVec[goodIdx])

    fieldArray = fill(NaN,length(prGrid),length(llGrid),4)

    for zVar in enumerate(zVars)
      (clVert, clHorz) = calcCorrLengths(zVar[2],obsLat=lati,obsLon=loni
      ,obsPres=prsi,presGrid=prGrid)

      if zVar[1] == 1
        reinterpolatedFieldsStruct.horzCorrLenTe = clHorz
        reinterpolatedFieldsStruct.vertCorrLenTe = clVert
      elseif zVar[1] == 2
        reinterpolatedFieldsStruct.horzCorrLenTr = clHorz
        reinterpolatedFieldsStruct.vertCorrLenTr = clVert
      elseif zVar[1] == 3
        reinterpolatedFieldsStruct.horzCorrLenSe = clHorz
        reinterpolatedFieldsStruct.vertCorrLenSe = clVert
      elseif zVar[1] == 4
        reinterpolatedFieldsStruct.horzCorrLenSr = clHorz
        reinterpolatedFieldsStruct.vertCorrLenSr = clVert
      end

    end

      for cruiseNo = 1:tBoxNum
        
        (ExcessTempVec,RedistTempVec) =  (ExcessTemp[:,:,cruiseNo][:],RedistTemp[:,:,cruiseNo][:])
        (ExcessSalVec,RedistSalVec) =  (ExcessSal[:,:,cruiseNo][:],RedistSal[:,:,cruiseNo][:])
        zVars = (ExcessTempVec[goodIdx],RedistTempVec[goodIdx]
              ,ExcessSalVec[goodIdx],RedistSalVec[goodIdx])

        for zVar in enumerate(zVars)
          if zVar[1] == 1
            clHorz = reinterpolatedFieldsStruct.horzCorrLenTe
            clVert = reinterpolatedFieldsStruct.vertCorrLenTe
          elseif zVar[1] == 2
            clHorz = reinterpolatedFieldsStruct.horzCorrLenTr
            clVert = reinterpolatedFieldsStruct.vertCorrLenTr
          elseif zVar[1] == 3
            clHorz = reinterpolatedFieldsStruct.horzCorrLenSe
            clVert = reinterpolatedFieldsStruct.vertCorrLenSe
          elseif zVar[1] == 4
            clHorz = reinterpolatedFieldsStruct.horzCorrLenSr
            clVert = reinterpolatedFieldsStruct.vertCorrLenSr
          end

          if horzCoordinate == "longitude"
            reinterpolatedField = easyDIVAGrid(variable=zVar[2],vertVar=prsi,latLon=loni
            ,horzCoordinate=horzCoordinate,meanValue="scalar",vertGrid=prGrid,horzGrid=llGrid
            ,horzScale=scaleHorz,vertScale=scaleVert,mask=mask,Epsilon=0.2,horzCorrLength=horzLenFactor*clHorz
            ,vertCorrLength = vertLenFactor*clVert)
          elseif horzCoordinate == "latitude"
            reinterpolatedField = easyDIVAGrid(variable=zVar[2],vertVar=prsi,latLon=lati
            ,horzCoordinate=horzCoordinate,meanValue="scalar",vertGrid=prGrid,horzGrid=llGrid
            ,horzScale=scaleHorz,vertScale=scaleVert,mask=mask,Epsilon=0.2,horzCorrLength=horzLenFactor*clHorz
            ,vertCorrLength = vertLenFactor*clVert)
          end
          fieldArray[:,:,zVar[1]] = reinterpolatedField
        end

        reinterpolatedFieldsStruct.ExcessTemp[:,:,cruiseNo] = fieldArray[:,:,1]
        reinterpolatedFieldsStruct.RedistTemp[:,:,cruiseNo] = fieldArray[:,:,2]
        reinterpolatedFieldsStruct.ExcessSalt[:,:,cruiseNo] = fieldArray[:,:,3]
        reinterpolatedFieldsStruct.RedistSalt[:,:,cruiseNo] = fieldArray[:,:,4]
      end

    return reinterpolatedFieldsStruct

  end

function relaxExcess(excessTemp::Matrix{Float64}, redistTemp::Matrix{Float64}
                    ;zDim = 1,scaleFactor::Float64=1.0,sigmoidOffset::Float64=0.5
                    ,numRepeats::Int64=10)

  ΔΘ_init = [excessTemp, redistTemp]
  (ΔΘe_new, ΔΘr_new) = (excessTemp, redistTemp)
  ΔΘ_tot = excessTemp + redistTemp
  ΔΘeScale = excessTempScale(excessTemp,zDimension=zDim)
  for i = 1:numRepeats
    ΔΘe_sym = excessTemp - ΔΘeScale
    ΔΘr_sym = redistTemp - ΔΘeScale
    ΔΘr_new = redistTemp - scaleFactor *ΔΘr_sym .*
     sigmoid.( abs.((excessTemp.*redistTemp).^4 ./ ΔΘ_tot) .- sigmoidOffset )
    ΔΘe_new = excessTemp - scaleFactor *ΔΘe_sym .*
     sigmoid.( abs.((excessTemp.*redistTemp).^4 ./ ΔΘ_tot) .- sigmoidOffset )
    ΔΘ_init = [ΔΘe_new, ΔΘr_new]
    ΔΘeScale = excessTempScale(ΔΘe_new,zDimension=zDim)
  end

  return (ΔΘe_new, ΔΘr_new)
end

function relaxExcess(excessTemp::Array{Float64, 3}, redistTemp::Array{Float64, 3}
                    ;zDim = 1,scaleFactor::Float64=1.0,sigmoidOffset::Float64=0.5
                    ,numRepeats::Int64=10)

  numYears = size(excessTemp,3)
  ΔΘe_new = fill(NaN,size(excessTemp,1),size(excessTemp,2),size(excessTemp,3))
  ΔΘr_new = fill(NaN,size(excessTemp,1),size(excessTemp,2),size(excessTemp,3))
  @threads for i = 1:numYears
    (ΔΘe_new[:,:,i],ΔΘr_new[:,:,i]) = relaxExcess(excessTemp[:,:,i],redistTemp[:,:,i]
                                                ,zDim=zDim,scaleFactor=scaleFactor
                                                ,sigmoidOffset=sigmoidOffset
                                                ,numRepeats=numRepeats)
  end
  return (ΔΘe_new, ΔΘr_new)
end


function postMaskPoles(excessTemp::Matrix{Float64}, redistTemp::Matrix{Float64}
                    ;scaleFactor::Number=1.0,sigmoidOffset::Number=0.5
                    ,power::Number=4,cutoffVal::Number=0.1)

  mask = fill(1.0,size(excessTemp))
  Δθ_tot = excessTemp + redistTemp

  SF = sigmoid.( abs.((excessTemp.*redistTemp).^power ./ Δθ_tot)*scaleFactor .- sigmoidOffset )
  SF[Δθ_tot.==0]  .= 0

  mask[SF .> cutoffVal] .= NaN


  p1 = heatmap(excessTemp,yflip=true,c=cgrad(:redblue,rev=true),clim=(-1,1),title="Raw Field")
  p2 = heatmap(mask .* excessTemp,yflip=true,c=cgrad(:redblue,rev=true),clim=(-1,1),title="Post identified mask")
  p3 = heatmap(SF,yflip=true,c=cgrad(:redblue,rev=true),clim=(0,1),title="Sigmoid scale factor")
  display(plot(p1,p2,p3,size=(800,500)))

  return mask
end

function postMaskPoles(excessTemp::Array{Float64, 3}, redistTemp::Array{Float64, 3}
                    ;scaleFactor::Number=1.0,sigmoidOffset::Number=0.5
                    ,power::Number=4,cutoffVal::Number=0.1)

  numYears = size(excessTemp,3)
  mask = fill(1.0,size(excessTemp))

  for i = 1:numYears
    mask[:,:,i] = postMaskPoles(excessTemp[:,:,i],redistTemp[:,:,i],scaleFactor=scaleFactor
                               ,sigmoidOffset=sigmoidOffset,cutoffVal=cutoffVal)
  end
  return mask
end

function TempDependentExcessRedistTempSalFromTempSalDIC(
                                     initTemp::Matrix{Float64},finalTemp::Matrix{Float64}
                                    ,initSal::Matrix{Float64},finalSal::Matrix{Float64}
                                    ,initDIC::Matrix{Float64},finalDIC::Matrix{Float64}
                                    ;zDimension::Int64=1
                                    ,α0= 0.025
                                    ,α1=-0.0005
                                    ,gaussianWidthParameter::Float64=0.0001)

    if size(initTemp) == size(finalTemp) == size(initDIC) == size(finalDIC) ==
       size(initSal) == size(finalSal)
      nothing
    else
      error("\"initTemp\", \"finalTemp\", \"initDIC\", \"finalDIC\"
            \"initSal\", \"finalSal\" must all be the same size")
    end


    ζ = centralDiff(initDIC,dim=zDimension)
    ξ = centralDiff(initTemp,dim=zDimension)
    η = centralDiff(initSal,dim=zDimension)

    κrT = ξ ./ ζ
    κrS = η ./ ζ
    τ = η ./ ξ
    α = initTemp .* α1 .+ α0
    @assert size(α) == size(κrT)

    weight1 = gaussian(ζ,σ = gaussianWidthParameter)
    weight2 = (weight1 .- 1.0) ./ (κrT - α)

    LENGTH_1 = size(initTemp,1)
    LENGTH_2 = size(initTemp,2)

    mat1 = fill(NaN,3,2,LENGTH_1,LENGTH_2)
    mat2 = fill(NaN,3,2,LENGTH_1,LENGTH_2)

    for i = 1:LENGTH_1, j = 1:LENGTH_2
      mat1[:,:,i,j] = [0       α[i,j]
                      ;1      -α[i,j]
                      ;τ[i,j] -α[i,j]*τ[i,j]]

      mat2[:,:,i,j] = [α[i,j]    -α[i,j]*κrT[i,j]
                      ;-κrT[i,j]  α[i,j]*κrT[i,j]
                      ;-κrS[i,j]  α[i,j]*κrS[i,j] ]
    end

    ΔΘ = finalTemp - initTemp
    ΔDIC = finalDIC - initDIC
    ΔSal = finalSal - initSal

    inputVectors = fill(NaN,2,LENGTH_1,LENGTH_2);
    outputVectors = fill(NaN,3,LENGTH_1,LENGTH_2);
    inputVectors[1,:,:] = ΔΘ
    inputVectors[2,:,:] = ΔDIC

    for i =1:LENGTH_1, j = 1:LENGTH_2
      outputVectors[:,i,j] = (weight1[i,j] * mat1[:,:,i,j] + weight2[i,j] * mat2[:,:,i,j]) * inputVectors[:,i,j]
    end

    ExcessTemperature = outputVectors[1,:,:]
    RedistTemperature = outputVectors[2,:,:]
    RedistSalinity    = outputVectors[3,:,:]
    ExcessSalinity    = ΔSal - RedistSalinity

    return ExcessTemperature, RedistTemperature, ExcessSalinity, RedistSalinity
 end

function TempDependentExcessRedistTempSalFromTempSalDIC(Temp::Array{Float64,3}
                                    ,Sal::Array{Float64,3},DIC::Array{Float64,3}
                                    ;zDimension::Int64=1
                                    ,tDimension::Int64=3
                                    ,α0= 0.025
                                    ,α1=-0.0005
                                    ,gaussianWidthParameter::Float64=0.0001)
    # Todo: check tDimension and reshape matrix if it's not 3

    if size(Temp) == size(DIC) == size(Sal)
      nothing
    else
      error("\"Temp\", \"Sal\" and \"DIC\" must be the same size")
    end

    initTemp = Temp[:,:,1]; initDIC = DIC[:,:,1]; initSal = Sal[:,:,1]
    ExcessTemperature = fill(NaN,size(Temp,1),size(Temp,2),size(Temp,3)-1)
    RedistTemperature = fill(NaN,size(Temp,1),size(Temp,2),size(Temp,3)-1)
    ExcessSalinity = fill(NaN,size(Sal,1),size(Sal,2),size(Sal,3)-1)
    RedistSalinity = fill(NaN,size(Sal,1),size(Sal,2),size(Sal,3)-1)

    NUM_REPEATS = size(ExcessTemperature,3)

    for i = 1:NUM_REPEATS
      finalTemp = Temp[:,:,i+1]; finalDIC = DIC[:,:,i+1]; finalSal = Sal[:,:,i+1]
      cruiseExcessTemp, cruiseRedistTemp, cruiseExcessSal, cruiseRedistSal =
      TempDependentExcessRedistTempSalFromTempSalDIC(initTemp,finalTemp,initSal,finalSal
                                 ,initDIC,finalDIC,zDimension=zDimension
                                 ,α0= α0
                                 ,α1= α1
                                 ,gaussianWidthParameter=gaussianWidthParameter)

      ExcessTemperature[:,:,i] = cruiseExcessTemp
      RedistTemperature[:,:,i] = cruiseRedistTemp
      ExcessSalinity[:,:,i] = cruiseExcessSal
      RedistSalinity[:,:,i] = cruiseRedistSal
    end

    return ExcessTemperature, RedistTemperature, ExcessSalinity, RedistSalinity

 end

function ExcessRedistTempSalFromTempSalDIC_BetaError(initTemp::Matrix{Float64},finalTemp::Matrix{Float64}
                                    ,initSal::Matrix{Float64},finalSal::Matrix{Float64}
                                    ,initDIC::Matrix{Float64},finalDIC::Matrix{Float64}
                                    ;transientValue::Float64=0.017
                                    ,zDimension::Int64=1
                                    ,gaussianWidthParameter::Float64=0.0001
                                    ,betaUncertainty::Float64=0.05
                                    ,maskLowerLim::Number=1)

    if size(initTemp) == size(finalTemp) == size(initDIC) == size(finalDIC) ==
       size(initSal) == size(finalSal)
      nothing
    else
      error("\"initTemp\", \"finalTemp\", \"initDIC\", \"finalDIC\"
            \"initSal\", \"finalSal\" must all be the same size")
    end


    ζ = centralDiff(initDIC,dim=zDimension)
    ξ = centralDiff(initTemp,dim=zDimension)
    η = centralDiff(initSal,dim=zDimension)

    κrT = Normal(1,betaUncertainty) |> x -> rand(x,size(initTemp)) |> x-> ξ ./ ζ .* x
    κrS = Normal(1,betaUncertainty) |> x -> rand(x,size(initTemp)) |> x-> η ./ ζ .* x
    τ =   Normal(1,betaUncertainty) |> x -> rand(x,size(initTemp)) |> x-> η ./ ξ .* x

    weight1 = gaussian(ζ,σ = gaussianWidthParameter)
    weight2 = (weight1 .- 1.0) ./ (κrT .- transientValue)

    LENGTH_1 = size(initTemp,1)
    LENGTH_2 = size(initTemp,2)

    mat1 = fill(NaN,3,2,LENGTH_1,LENGTH_2)
    mat2 = fill(NaN,3,2,LENGTH_1,LENGTH_2)

    for i = 1:LENGTH_1, j = 1:LENGTH_2
      mat1[:,:,i,j] = [0              transientValue
                    ;1              -transientValue
                    ;τ[i,j]              -transientValue*τ[i,j]]

      mat2[:,:,i,j] = [transientValue -transientValue*κrT[i,j]
                    ;-κrT[i,j]         transientValue*κrT[i,j]
                    ;-κrS[i,j]         transientValue*κrS[i,j] ]
    end

    ΔΘ = finalTemp - initTemp
    ΔDIC = finalDIC - initDIC
    ΔSal = finalSal - initSal

    inputVectors = fill(NaN,2,LENGTH_1,LENGTH_2);
    outputVectors = fill(NaN,3,LENGTH_1,LENGTH_2);
    inputVectors[1,:,:] = ΔΘ
    inputVectors[2,:,:] = ΔDIC

    for i =1:LENGTH_1, j = 1:LENGTH_2
      outputVectors[:,i,j] = (weight1[i,j] * mat1[:,:,i,j] + weight2[i,j] * mat2[:,:,i,j]) * inputVectors[:,i,j]
    end

    ExcessTemperature = outputVectors[1,:,:]
    RedistTemperature = outputVectors[2,:,:]
    RedistSalinity    = outputVectors[3,:,:]
    ExcessSalinity    = ΔSal - RedistSalinity

    αT = transientValue

    λ = (αT .* ( κrT .+ 1 ) + sqrt.(Complex.( αT .^2 .* (κrT .+ 1) .^2 - 4 .* αT .* κrT .* (κrT .+ αT)))) ./ (-2 .* (κrT .- αT))
    SF = min.(1,max.(1 .- log.(abs.(λ)),0))
    SF[SF .< maskLowerLim] .= NaN

    return ExcessTemperature .* SF, RedistTemperature .* SF, ExcessSalinity .* SF, RedistSalinity .* SF
 end



function ExcessRedistTempSalFromTempSalDIC_BetaErrorTDependent(
                                    initTemp::Matrix{Float64},finalTemp::Matrix{Float64}
                                    ,initSal::Matrix{Float64},finalSal::Matrix{Float64}
                                    ,initDIC::Matrix{Float64},finalDIC::Matrix{Float64}
                                    ;zDimension::Int64=1
                                    ,gaussianWidthParameter::Float64=0.005
                                    ,betaUncertainty::Float64=0.02
                                    ,α0::Float64=0.0248
                                    ,α1::Float64=-0.000505
                                    ,maskLowerLim::Number=1)

    if size(initTemp) == size(finalTemp) == size(initDIC) == size(finalDIC) ==
       size(initSal) == size(finalSal)
      nothing
    else
      error("\"initTemp\", \"finalTemp\", \"initDIC\", \"finalDIC\"
            \"initSal\", \"finalSal\" must all be the same size")
    end


    ζ = centralDiff(initDIC,dim=zDimension)
    ξ = centralDiff(initTemp,dim=zDimension)
    η = centralDiff(initSal,dim=zDimension)

    κrT = Normal(1,betaUncertainty) |> x -> rand(x,size(initTemp)) |> x-> ξ ./ ζ .* x
    κrS = Normal(1,betaUncertainty) |> x -> rand(x,size(initTemp)) |> x-> η ./ ζ .* x
    τ =   Normal(1,betaUncertainty) |> x -> rand(x,size(initTemp)) |> x-> η ./ ξ .* x

    α = initTemp .* α1 .+ α0
    @assert size(α) == size(κrT)

    weight1 = gaussian(ζ,σ = gaussianWidthParameter)
    weight2 = (weight1 .- 1.0) ./ (κrT - α)

    LENGTH_1 = size(initTemp,1)
    LENGTH_2 = size(initTemp,2)

    mat1 = fill(NaN,3,2,LENGTH_1,LENGTH_2)
    mat2 = fill(NaN,3,2,LENGTH_1,LENGTH_2)

    for i = 1:LENGTH_1, j = 1:LENGTH_2
      mat1[:,:,i,j] = [0       α[i,j]
                      ;1      -α[i,j]
                      ;τ[i,j] -α[i,j]*τ[i,j]]

      mat2[:,:,i,j] = [α[i,j]    -α[i,j]*κrT[i,j]
                      ;-κrT[i,j]  α[i,j]*κrT[i,j]
                      ;-κrS[i,j]  α[i,j]*κrS[i,j] ]
    end

    ΔΘ = finalTemp - initTemp
    ΔDIC = finalDIC - initDIC
    ΔSal = finalSal - initSal

    inputVectors = fill(NaN,2,LENGTH_1,LENGTH_2);
    outputVectors = fill(NaN,3,LENGTH_1,LENGTH_2);
    inputVectors[1,:,:] = ΔΘ
    inputVectors[2,:,:] = ΔDIC

    for i =1:LENGTH_1, j = 1:LENGTH_2
      outputVectors[:,i,j] = (weight1[i,j] * mat1[:,:,i,j] + weight2[i,j] * mat2[:,:,i,j]) * inputVectors[:,i,j]
    end

    ExcessTemperature = outputVectors[1,:,:]
    RedistTemperature = outputVectors[2,:,:]
    RedistSalinity    = outputVectors[3,:,:]
    ExcessSalinity    = ΔSal - RedistSalinity

    αT = initTemp .* α1 .+ α0
    βT = ξ ./ ζ

    λ = (αT .* ( βT .+ 1 ) + sqrt.(Complex.( αT .^2 .* (βT .+ 1) .^2 - 4 .* αT .* βT .* (βT .+ αT)))) ./ (-2 .* (βT .- αT))

    SF = min.(1,max.(1 .- log.(abs.(λ)),0))
    SF[SF .< maskLowerLim] .= NaN

    return ExcessTemperature .* SF, RedistTemperature .* SF, ExcessSalinity .* SF, RedistSalinity .* SF
 end